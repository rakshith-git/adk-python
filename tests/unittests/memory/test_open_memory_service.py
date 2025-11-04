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

from unittest.mock import MagicMock
from unittest.mock import patch

from google.adk.events.event import Event
from google.adk.memory.open_memory_service import (
    OpenMemoryService,
    OpenMemoryServiceConfig,
)
from google.adk.sessions.session import Session
from google.genai import types
import pytest

MOCK_APP_NAME = 'test-app'
MOCK_USER_ID = 'test-user'
MOCK_SESSION_ID = 'session-1'

MOCK_SESSION = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
    events=[
        Event(
            id='event-1',
            invocation_id='inv-1',
            author='user',
            timestamp=12345,
            content=types.Content(parts=[types.Part(text='Hello, I like Python.')]),
        ),
        Event(
            id='event-2',
            invocation_id='inv-2',
            author='model',
            timestamp=12346,
            content=types.Content(
                parts=[types.Part(text='Python is a great programming language.')]
            ),
        ),
        # Empty event, should be ignored
        Event(
            id='event-3',
            invocation_id='inv-3',
            author='user',
            timestamp=12347,
        ),
        # Function call event, should be ignored
        Event(
            id='event-4',
            invocation_id='inv-4',
            author='agent',
            timestamp=12348,
            content=types.Content(
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(name='test_function')
                    )
                ]
            ),
        ),
    ],
)

MOCK_SESSION_WITH_EMPTY_EVENTS = Session(
    app_name=MOCK_APP_NAME,
    user_id=MOCK_USER_ID,
    id=MOCK_SESSION_ID,
    last_update_time=1000,
)


@pytest.fixture
def mock_openmemory_client():
  """Mock OpenMemory SDK client for testing."""
  with patch('google.adk.memory.open_memory_service.OpenMemory') as mock_client_constructor:
    mock_client = MagicMock()
    mock_client.add = MagicMock()
    mock_client.query = MagicMock()
    mock_client_constructor.return_value = mock_client
    yield mock_client


@pytest.fixture
def memory_service(mock_openmemory_client):
  """Create OpenMemoryService instance for testing."""
  return OpenMemoryService(base_url='http://localhost:3000', api_key='test-key')


@pytest.fixture
def memory_service_with_config(mock_openmemory_client):
  """Create OpenMemoryService with custom config."""
  config = OpenMemoryServiceConfig(
      search_top_k=5,
      user_content_salience=0.9,
      model_content_salience=0.6
  )
  return OpenMemoryService(
      base_url='http://localhost:3000',
      api_key='test-key',
      config=config
  )


class TestOpenMemoryServiceConfig:
  """Tests for OpenMemoryServiceConfig."""

  def test_default_config(self):
    """Test default configuration values."""
    config = OpenMemoryServiceConfig()
    assert config.search_top_k == 10
    assert config.timeout == 30.0
    assert config.user_content_salience == 0.8
    assert config.model_content_salience == 0.7
    assert config.default_salience == 0.6
    assert config.enable_metadata_tags is True

  def test_custom_config(self):
    """Test custom configuration values."""
    config = OpenMemoryServiceConfig(
        search_top_k=20,
        timeout=10.0,
        user_content_salience=0.9,
        model_content_salience=0.75,
        default_salience=0.5,
        enable_metadata_tags=False
    )
    assert config.search_top_k == 20
    assert config.timeout == 10.0
    assert config.user_content_salience == 0.9
    assert config.model_content_salience == 0.75
    assert config.default_salience == 0.5
    assert config.enable_metadata_tags is False

  def test_config_validation_search_top_k(self):
    """Test search_top_k validation."""
    with pytest.raises(Exception):  # Pydantic validation error
      OpenMemoryServiceConfig(search_top_k=0)

    with pytest.raises(Exception):
      OpenMemoryServiceConfig(search_top_k=101)


class TestOpenMemoryService:
  """Tests for OpenMemoryService."""

  @pytest.mark.asyncio
  async def test_add_session_to_memory_success(self, memory_service, mock_openmemory_client):
    """Test successful addition of session memories."""
    mock_openmemory_client.add.return_value = {"id": "mem-1"}

    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should make 2 add calls (one per valid event)
    assert mock_openmemory_client.add.call_count == 2

    # Check first call (user event)
    call_args = mock_openmemory_client.add.call_args_list[0]
    request_data = call_args[0][0]
    assert request_data['content'] == 'Hello, I like Python.'
    assert 'session:session-1' in request_data['tags']
    assert request_data['metadata']['author'] == 'user'
    assert request_data['salience'] == 0.8  # User content salience

    # Check second call (model event)
    call_args = mock_openmemory_client.add.call_args_list[1]
    request_data = call_args[0][0]
    assert request_data['content'] == 'Python is a great programming language.'
    assert request_data['metadata']['author'] == 'model'
    assert request_data['salience'] == 0.7  # Model content salience

  @pytest.mark.asyncio
  async def test_add_session_filters_empty_events(
      self, memory_service, mock_openmemory_client
  ):
    """Test that events without content are filtered out."""
    await memory_service.add_session_to_memory(MOCK_SESSION_WITH_EMPTY_EVENTS)

    # Should make 0 add calls (no valid events)
    assert mock_openmemory_client.add.call_count == 0

  @pytest.mark.asyncio
  async def test_add_session_uses_config_salience(
      self, memory_service_with_config, mock_openmemory_client
  ):
    """Test that salience values from config are used."""
    mock_openmemory_client.add.return_value = {"id": "mem-1"}

    await memory_service_with_config.add_session_to_memory(MOCK_SESSION)

    # Check that custom salience values are used
    call_args = mock_openmemory_client.add.call_args_list[0]
    request_data = call_args[0][0]
    assert request_data['salience'] == 0.9  # Custom user salience

    call_args = mock_openmemory_client.add.call_args_list[1]
    request_data = call_args[0][0]
    assert request_data['salience'] == 0.6  # Custom model salience

  @pytest.mark.asyncio
  async def test_add_session_without_metadata_tags(
      self, mock_openmemory_client
  ):
    """Test adding memories without metadata tags."""
    config = OpenMemoryServiceConfig(enable_metadata_tags=False)
    memory_service = OpenMemoryService(
        base_url='http://localhost:3000', config=config
    )
    mock_openmemory_client.add.return_value = {"id": "mem-1"}

    await memory_service.add_session_to_memory(MOCK_SESSION)

    call_args = mock_openmemory_client.add.call_args_list[0]
    request_data = call_args[0][0]
    assert 'tags' not in request_data

  @pytest.mark.asyncio
  async def test_add_session_error_handling(self, memory_service, mock_openmemory_client):
    """Test error handling during memory addition."""
    mock_openmemory_client.add.side_effect = Exception('API Error')

    # Should not raise exception, just log error
    await memory_service.add_session_to_memory(MOCK_SESSION)

    # Should still attempt to make add calls
    assert mock_openmemory_client.add.call_count == 2

  @pytest.mark.asyncio
  async def test_search_memory_success(self, memory_service, mock_openmemory_client):
    """Test successful memory search."""
    mock_openmemory_client.query.return_value = {
        'items': [
            {
                'content': 'Python is great',
                'metadata': {
                    'author': 'user',
                    'timestamp': 12345,
                    'user_id': MOCK_USER_ID,
                    'app_name': MOCK_APP_NAME,
                }
            },
            {
                'content': 'I like programming',
                'metadata': {
                    'author': 'model',
                    'timestamp': 12346,
                    'user_id': MOCK_USER_ID,
                    'app_name': MOCK_APP_NAME,
                }
            }
        ]
    }

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='Python programming'
    )

    # Verify API call
    call_args = mock_openmemory_client.query.call_args
    request_data = call_args[0][0]
    assert request_data['query'] == 'Python programming'
    assert request_data['top_k'] == 10
    assert 'tags' in request_data['filter']

    # Verify results
    assert len(result.memories) == 2
    assert result.memories[0].content.parts[0].text == 'Python is great'
    assert result.memories[0].author == 'user'
    assert result.memories[1].content.parts[0].text == 'I like programming'
    assert result.memories[1].author == 'model'

  @pytest.mark.asyncio
  async def test_search_memory_applies_filters(
      self, memory_service, mock_openmemory_client
  ):
    """Test that app_name/user_id filters are applied."""
    mock_openmemory_client.query.return_value = {
        'items': [
            {
                'content': 'Python is great',
                'metadata': {
                    'author': 'user',
                    'timestamp': 12345,
                    'user_id': 'different-user',  # Different user
                    'app_name': MOCK_APP_NAME,
                }
            },
            {
                'content': 'I like programming',
                'metadata': {
                    'author': 'model',
                    'timestamp': 12346,
                    'user_id': MOCK_USER_ID,  # Correct user
                    'app_name': MOCK_APP_NAME,
                }
            }
        ]
    }

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='test query'
    )

    # Should filter out the different user's result
    assert len(result.memories) == 1
    assert result.memories[0].content.parts[0].text == 'I like programming'

  @pytest.mark.asyncio
  async def test_search_memory_respects_top_k(
      self, memory_service_with_config, mock_openmemory_client
  ):
    """Test that config.search_top_k is used."""
    mock_openmemory_client.query.return_value = {'items': []}

    await memory_service_with_config.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='test query'
    )

    call_args = mock_openmemory_client.query.call_args
    request_data = call_args[0][0]
    assert request_data['top_k'] == 5  # Custom config value

  @pytest.mark.asyncio
  async def test_search_memory_error_handling(
      self, memory_service, mock_openmemory_client
  ):
    """Test graceful error handling during memory search."""
    mock_openmemory_client.query.side_effect = Exception('API Error')

    result = await memory_service.search_memory(
        app_name=MOCK_APP_NAME,
        user_id=MOCK_USER_ID,
        query='test query'
    )

    # Should return empty results on error
    assert len(result.memories) == 0

  @pytest.mark.asyncio
  async def test_close_client(self, memory_service):
    """Test client cleanup (SDK handles automatically)."""
    # SDK handles cleanup automatically, so close() is a no-op
    await memory_service.close()
    # Should complete without error

