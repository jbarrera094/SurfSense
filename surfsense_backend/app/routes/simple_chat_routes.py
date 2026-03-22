"""
Simplified chat endpoint.

POST /simple-chat?search_space_id=1&chat_id={optional}&mentioned_document_ids=1&mentioned_document_ids=2
Body: { "user_query": "..." }
Response: { "ai_response": "...", "chat_id": 123 }

If chat_id is provided the existing thread is reused.
If chat_id is omitted a new thread is created automatically.
The AI response is collected by consuming the SSE stream internally,
so the client receives a plain JSON response (no streaming required).

A fixed set of heavyweight / media tools is always disabled for this
lightweight endpoint so responses stay fast and text-only.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db import (
    ChatVisibility,
    NewChatMessage,
    NewChatMessageRole,
    NewChatThread,
    Permission,
    SearchSpace,
    User,
    async_session_maker,
    get_async_session,
)
from app.routes.new_chat_routes import check_thread_access
from app.tasks.chat.stream_new_chat import stream_new_chat
from app.users import current_active_user
from app.utils.rbac import check_permission

_logger = logging.getLogger(__name__)

router = APIRouter()

# Tools that are always disabled for the simple-chat endpoint.
# TODO: Fix when I don't send chat_id then the endpoint create one chat from scrash, right? 
# but this new chat doesn't take into consideration the document ID I passed to it. So help me to fixed that
# temporal removed search_knowledge_base tool
_DISABLED_TOOLS: list[str] = [
    "generate_podcast",
    "generate_report",
    "link_preview",
    "display_image",
    "generate_image",
    "scrape_webpage",
    "search_surfsense_docs"
]


# =============================================================================
# Schemas
# =============================================================================


class SimpleChatRequest(BaseModel):
    user_query: str


class SimpleChatResponse(BaseModel):
    ai_response: str
    chat_id: int


# =============================================================================
# Endpoint
# =============================================================================


@router.post("/simple-chat", response_model=SimpleChatResponse)
async def simple_chat(
    request: SimpleChatRequest,
    search_space_id: int,
    chat_id: int | None = None,
    mentioned_document_ids: Annotated[list[int] | None, Query()] = None,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    """
    Send a message and receive the full AI response as plain JSON.

    - If chat_id is provided the message is appended to that existing thread.
    - If chat_id is omitted a new private thread is created and its id is
      returned so the caller can continue the conversation later.
    - Pass mentioned_document_ids as repeated query params to inject specific
      documents as context: ?mentioned_document_ids=1&mentioned_document_ids=2
    - A fixed set of tools (podcast, report, image, scraping, KB search, etc.)
      is always disabled to keep responses fast and text-only.

    Requires CHATS_CREATE permission on the search space.
    """
    try:
        await check_permission(
            session,
            user,
            search_space_id,
            Permission.CHATS_CREATE.value,
            "You don't have permission to chat in this search space",
        )

        # Resolve or create the thread
        if chat_id is not None:
            result = await session.execute(
                select(NewChatThread).filter(NewChatThread.id == chat_id)
            )
            thread = result.scalars().first()

            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")

            if thread.search_space_id != search_space_id:
                raise HTTPException(
                    status_code=400,
                    detail="Thread does not belong to the specified search space",
                )

            await check_thread_access(session, thread, user)
        else:
            thread = NewChatThread(
                title="New Chat",
                archived=False,
                visibility=ChatVisibility.SEARCH_SPACE,
                search_space_id=search_space_id,
                created_by_id=user.id,
                updated_at=datetime.now(UTC),
            )
            session.add(thread)
            await session.flush()
            await session.commit()
            await session.refresh(thread)
            chat_id = thread.id

        # Load the search space to determine LLM config
        search_space_result = await session.execute(
            select(SearchSpace).filter(SearchSpace.id == search_space_id)
        )
        search_space = search_space_result.scalars().first()

        if not search_space:
            raise HTTPException(status_code=404, detail="Search space not found")

        llm_config_id = (
            search_space.agent_llm_id if search_space.agent_llm_id is not None else -1
        )

        # Save the user message before releasing the session.
        content_parts: list[dict] = [{"type": "text", "text": request.user_query}]
        if mentioned_document_ids:
            for doc_id in mentioned_document_ids:
                content_parts.append({"type": "document_mention", "document_id": doc_id})

        user_msg = NewChatMessage(
            thread_id=chat_id,
            role=NewChatMessageRole.USER,
            content=content_parts,
            author_id=user.id,
        )
        session.add(user_msg)
        thread.updated_at = datetime.now(UTC)

        # Release the DB session before consuming the streaming generator,
        # same pattern as the existing /new_chat endpoint.
        await session.commit()
        await session.close()

        # Consume the SSE stream and accumulate the plain text response
        ai_text = ""
        async for chunk in stream_new_chat(
            user_query=request.user_query,
            search_space_id=search_space_id,
            chat_id=chat_id,
            user_id=str(user.id),
            llm_config_id=llm_config_id,
            mentioned_document_ids=mentioned_document_ids or [],
            needs_history_bootstrap=thread.needs_history_bootstrap,
            thread_visibility=thread.visibility,
            current_user_display_name=user.display_name or "A team member",
            disabled_tools=_DISABLED_TOOLS,
        ):
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    try:
                        event = json.loads(line[6:])
                        if event.get("type") == "text-delta":
                            ai_text += event.get("delta", "")
                    except (json.JSONDecodeError, TypeError):
                        pass

        # Persist the AI response message using a fresh session (the original
        # session was closed before streaming began).
        async with async_session_maker() as ai_session:
            ai_msg = NewChatMessage(
                thread_id=chat_id,
                role=NewChatMessageRole.ASSISTANT,
                content=[{"type": "text", "text": ai_text}],
                author_id=None,
            )
            ai_session.add(ai_msg)
            thread_result = await ai_session.execute(
                select(NewChatThread).filter(NewChatThread.id == chat_id)
            )
            updated_thread = thread_result.scalars().first()
            if updated_thread:
                updated_thread.updated_at = datetime.now(UTC)
            await ai_session.commit()

        return SimpleChatResponse(ai_response=ai_text, chat_id=chat_id)

    except HTTPException:
        raise
    except Exception as e:
        _logger.exception("Unexpected error in simple_chat")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {e!s}",
        ) from None
