import sys
import os
import asyncio
import pytest
from qdrant_client import models


class FakeOperation:
    def __init__(self, operation_id="op-1"):
        self.operation_id = operation_id


class FakeHit:
    def __init__(self, _id="123", score=0.9, payload=None):
        self.id = _id
        self.score = score
        self.payload = payload or {"file_path": "foo.py", "content": "print('x')"}


class FakeAsyncQdrantClient:
    def __init__(self, *args, **kwargs):
        pass

    async def get_collections(self):
        class _Resp:
            class _Result:
                def __init__(self):
                    self.collections = []
            def __init__(self):
                self.result = self._Result()
        return _Resp()

    async def search(self, **kwargs):
        return [FakeHit()]

    async def delete(self, **kwargs):
        return FakeOperation()



@pytest.mark.asyncio
async def test_search_vectors_returns_dict(monkeypatch):
    # Arrange: use real qdrant_client.models
    # Ensure module path can find services
    sys.path.insert(0, os.path.abspath("neural-tools/src"))
    from servers.services import qdrant_service as qs

    monkeypatch.setattr(qs, "AsyncQdrantClient", FakeAsyncQdrantClient)

    svc = qs.QdrantService(project_name="test")
    init = await svc.initialize()
    assert init.get("success") is True

    # Act
    results = await svc.search_vectors(
        collection_name="test_collection",
        query_vector=[0.0] * 3,
        limit=5,
    )

    # Assert
    assert isinstance(results, list)
    assert isinstance(results[0], dict)
    assert set(results[0].keys()) == {"id", "score", "payload"}


@pytest.mark.asyncio
async def test_delete_points_by_ids_and_filter(monkeypatch):
    # Arrange
    # Arrange: use real qdrant_client.models
    sys.path.insert(0, os.path.abspath("neural-tools/src"))
    from servers.services import qdrant_service as qs

    monkeypatch.setattr(qs, "AsyncQdrantClient", FakeAsyncQdrantClient)

    svc = qs.QdrantService(project_name="test")
    await svc.initialize()

    # By IDs using real PointIdsList
    res_ids = await svc.delete_points(
        collection_name="test_collection",
        points_selector=models.PointIdsList(points=["a", "b"]),
    )
    assert res_ids.get("status") == "success"

    # By filter using real FilterSelector
    res_filter = await svc.delete_points(
        collection_name="test_collection",
        filter_conditions=models.FilterSelector(filter=models.Filter(must=[])),
    )
    assert res_filter.get("status") == "success"
