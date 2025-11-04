# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import re
from typing import Optional
from typing import TYPE_CHECKING

import httpx
from google.genai import types
from pydantic import BaseModel
from pydantic import Field
from typing_extensions import override

from . import _utils
from .base_memory_service import BaseMemoryService
from .base_memory_service import SearchMemoryResponse
from .memory_entry import MemoryEntry

if TYPE_CHECKING:
  from ..sessions.session import Session

logger = logging.getLogger('google_adk.' + __name__)

class OpenMemoryService(BaseMemoryService):
  """Memory service implementation using OpenMemory.
  See https://openmemory.cavira.app/ for more information.

  OpenMemory provides hierarchical memory decomposition with multi-sector
  embeddings, graceful decay curves, and automatic reinforcement for AI agents.

  Implementation Note:
      This service uses direct HTTP requests via httpx for all operations.
      This ensures full control over the request format, particularly for
      passing `user_id` as a top-level field (required for server-side
      filtering in queries).

  Example:
      ```python
      from google.adk.memory import OpenMemoryService, OpenMemoryServiceConfig

      # Basic usage with defaults
      memory_service = OpenMemoryService(
          base_url="http://localhost:3000"
      )

      # Custom configuration
      config = OpenMemoryServiceConfig(
          search_top_k=20,
          user_content_salience=0.9,
          model_content_salience=0.75
      )
      memory_service = OpenMemoryService(
          base_url="http://localhost:3000",
          api_key="my-api-key",
          config=config
      )
      ```
  """

  def __init__(
      self,
      base_url: str = "http://localhost:3000",
      api_key: Optional[str] = None,
      config: Optional[OpenMemoryServiceConfig] = None,
  ):
    """Initializes the OpenMemory service.

    Args:
        base_url: Base URL of the OpenMemory instance (default: localhost:3000).
        api_key: API key for authentication (optional, only if server requires).
        config: OpenMemoryServiceConfig instance for customizing behavior. If
            None, uses default configuration.
    """
    self._base_url = base_url.rstrip('/')
    self._api_key = api_key
    self._config = config or OpenMemoryServiceConfig()

  def _determine_salience(self, author: Optional[str]) -> float:
    """Determine salience value based on content author.

    Args:
        author: The author of the content (e.g., 'user', 'model', 'system').

    Returns:
        Salience value between 0.0 and 1.0.
    """
    if not author:
      return self._config.default_salience

    author_lower = author.lower()
    if author_lower == "user":
      return self._config.user_content_salience
    elif author_lower == "model":
      return self._config.model_content_salience
    else:
      return self._config.default_salience

  def _prepare_memory_data(
      self, event, content_text: str, session
  ) -> dict:
    """Prepare memory data structure for OpenMemory API.

    Embeds author and timestamp directly in the content string so they
    are available during search without needing additional API calls.
    This avoids N+1 query problem when searching.

    Args:
        event: The event to create memory from.
        content_text: Extracted text content.
        session: The session containing the event.

    Returns:
        Dictionary with memory data formatted for OpenMemory API.
    """
    # Format timestamp for display
    timestamp_str = None
    if event.timestamp:
      timestamp_str = _utils.format_timestamp(event.timestamp)
    
    # Embed author and timestamp in content for retrieval during search
    # Format: [Author: user, Time: 2025-11-04T10:32:01] Content text
    enriched_content = content_text
    metadata_parts = []
    if event.author:
      metadata_parts.append(f"Author: {event.author}")
    if timestamp_str:
      metadata_parts.append(f"Time: {timestamp_str}")
    
    if metadata_parts:
      metadata_prefix = "[" + ", ".join(metadata_parts) + "] "
      enriched_content = metadata_prefix + content_text
    
    # Store metadata for filtering and tracking
    metadata = {
        "app_name": session.app_name,
        "user_id": session.user_id,
        "session_id": session.id,
        "event_id": event.id,
        "invocation_id": event.invocation_id,
        "author": event.author,
        "timestamp": event.timestamp,
        "source": "adk_session"
    }
    
    memory_data = {
        "content": enriched_content,
        "metadata": metadata,
        "salience": self._determine_salience(event.author)
    }

    if self._config.enable_metadata_tags:
      memory_data["tags"] = [
          f"session:{session.id}",
          f"app:{session.app_name}",
          f"author:{event.author}" if event.author else None
      ]
      # Remove None values
      memory_data["tags"] = [t for t in memory_data["tags"] if t]

    return memory_data

  @override
  async def add_session_to_memory(self, session: Session):
    """Add a session's events to OpenMemory.

    Processes all events in the session, filters out empty content,
    and adds meaningful memories to OpenMemory with appropriate metadata.

    Args:
        session: The session containing events to add to memory.
    """
    memories_added = 0

    # Create HTTP client once for all events to improve performance
    async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
      headers = {"Content-Type": "application/json"}
      if self._api_key:
        headers["Authorization"] = f"Bearer {self._api_key}"

      for event in session.events:
        content_text = _extract_text_from_event(event)
        if not content_text:
          continue

        memory_data = self._prepare_memory_data(event, content_text, session)

        try:
          # Use direct HTTP to pass user_id as top-level field 
          # This ensures server-side filtering works correctly
          
          # Include user_id as separate field for database storage and filtering
          payload = {
              "content": memory_data["content"],
              "tags": memory_data.get("tags", []),
              "metadata": memory_data.get("metadata", {}),
              "salience": memory_data.get("salience", 0.5),
              "user_id": session.user_id  # Separate field for DB column
          }
          
          response = await http_client.post(
              f"{self._base_url}/memory/add",
              json=payload,
              headers=headers
          )
          response.raise_for_status()
          
          memories_added += 1
          logger.debug("Added memory for event %s", event.id)
        except Exception as e:
          logger.error("Failed to add memory for event %s: %s", event.id, e)

    logger.info(
        "Added %d memories from session %s", memories_added, session.id
    )

  def _build_search_payload(
      self, app_name: str, user_id: str, query: str
  ) -> dict:
    """Build search payload for OpenMemory query API.

    Args:
        app_name: The application name to filter by.
        user_id: The user ID to filter by.
        query: The search query string.

    Returns:
        Dictionary with query parameters formatted for HTTP API.
    """
    payload = {
        "query": query,
        "k": self._config.search_top_k,  # Backend expects 'k', not 'top_k'
        "filter": {}
    }

    # Always filter by user_id for multi-user isolation
    payload["filter"]["user_id"] = user_id

    # Add tag-based filtering if enabled
    if self._config.enable_metadata_tags:
      payload["filter"]["tags"] = [f"app:{app_name}"]

    return payload

  def _convert_to_memory_entry(self, result: dict) -> Optional[MemoryEntry]:
    """Convert OpenMemory result to MemoryEntry.

    Extracts author and timestamp from the enriched content format:
    [Author: user, Time: 2025-11-04T10:32:01] Content text

    Args:
        result: OpenMemory search result (match from query or full memory).

    Returns:
        MemoryEntry or None if conversion fails.
    """
    try:
      raw_content = result["content"]
      author = None
      timestamp = None
      clean_content = raw_content
      
      # Parse enriched content format: [Author: X, Time: Y] Content
      match = re.match(r'^\[([^\]]+)\]\s+(.*)', raw_content, re.DOTALL)
      if match:
        metadata_str = match.group(1)
        clean_content = match.group(2)
        
        # Extract author
        author_match = re.search(r'Author:\s*([^,\]]+)', metadata_str)
        if author_match:
          author = author_match.group(1).strip()
        
        # Extract timestamp
        time_match = re.search(r'Time:\s*([^,\]]+)', metadata_str)
        if time_match:
          timestamp = time_match.group(1).strip()
      
      # Create content with clean text (without metadata prefix)
      content = types.Content(parts=[types.Part(text=clean_content)])

      return MemoryEntry(
          content=content,
          author=author,
          timestamp=timestamp
      )
    except (KeyError, ValueError) as e:
      logger.debug("Failed to convert result to MemoryEntry: %s", e)
      return None

  @override
  async def search_memory(
      self, *, app_name: str, user_id: str, query: str
  ) -> SearchMemoryResponse:
    """Search for memories using OpenMemory's query API.

    Queries OpenMemory with the search string and filters results
    by app_name and user_id using tags (if metadata tagging is enabled).

    Args:
        app_name: The application name to filter memories by.
        user_id: The user ID to filter memories by.
        query: The search query string.

    Returns:
        SearchMemoryResponse containing matching memories.
    """
    try:
      search_payload = self._build_search_payload(app_name, user_id, query)

      # Use direct HTTP call for query since SDK v0.2.0 filters don't work properly
      # The SDK doesn't correctly pass filter parameters to the backend
      memories = []
      
      async with httpx.AsyncClient(timeout=self._config.timeout) as http_client:
        # Query for matching memories
        headers = {"Content-Type": "application/json"}
        if self._api_key:
          headers["Authorization"] = f"Bearer {self._api_key}"
        
        # Debug: Log the exact payload being sent
        logger.debug("Query payload: %s", search_payload)
        
        response = await http_client.post(
            f"{self._base_url}/memory/query",
            json=search_payload,
            headers=headers
        )
        response.raise_for_status()
        result = response.json()
        
        # Debug: Log response summary
        logger.debug("Query returned %d matches", len(result.get("matches", [])))
        
        # Backend returns 'matches' with content already including author/timestamp
        # No need for additional API calls - parse from enriched content format
        for match in result.get("matches", []):
          memory_entry = self._convert_to_memory_entry(match)
          if memory_entry:
            memories.append(memory_entry)

      logger.info("Found %d memories for query: '%s'", len(memories), query)
      return SearchMemoryResponse(memories=memories)

    except Exception as e:
      logger.error("Failed to search memories: %s", e)
      return SearchMemoryResponse(memories=[])

  async def close(self):
    """Close the memory service and cleanup resources.
    
    This method is provided for API consistency. Since httpx.AsyncClient
    is used as a context manager in all operations, cleanup is handled
    automatically. This method is a no-op and can be safely called or omitted.
    """
    pass


class OpenMemoryServiceConfig(BaseModel):
  """Configuration for OpenMemory service behavior.

  Attributes:
      search_top_k: Maximum number of memories to retrieve per search.
      timeout: Request timeout in seconds.
      user_content_salience: Salience for user-authored content (0.0-1.0).
      model_content_salience: Salience for model-generated content (0.0-1.0).
      default_salience: Default salience value for memories (0.0-1.0).
      enable_metadata_tags: Include session/app tags in memories.
  """

  search_top_k: int = Field(default=10, ge=1, le=100)
  timeout: float = Field(default=30.0, gt=0.0)
  user_content_salience: float = Field(default=0.8, ge=0.0, le=1.0)
  model_content_salience: float = Field(default=0.7, ge=0.0, le=1.0)
  default_salience: float = Field(default=0.6, ge=0.0, le=1.0)
  enable_metadata_tags: bool = Field(default=True)

def _extract_text_from_event(event) -> str:
  """Extracts text content from an event's content parts.

  Args:
      event: The event to extract text from.

  Returns:
      Combined text from all text parts, or empty string if none found.
  """
  if not event.content or not event.content.parts:
    return ''

  text_parts = [part.text for part in event.content.parts if part.text]
  return ' '.join(text_parts)
