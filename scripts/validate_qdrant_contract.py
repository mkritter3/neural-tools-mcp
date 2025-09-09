#!/usr/bin/env python3
"""
Local validation of QdrantService contract without external services.
It injects a fake AsyncQdrantClient and minimal qdrant_client.models to
exercise initialize(), search_vectors(), and delete_points().
"""

import sys
import os
import asyncio
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



async def main():
    # Ensure services path
    sys.path.insert(0, os.path.abspath("neural-tools/src"))
    # Inject fake async client only; use real qdrant_client.models
    from servers.services import qdrant_service as qs
    qs.AsyncQdrantClient = FakeAsyncQdrantClient

    svc = qs.QdrantService(project_name="test")
    init = await svc.initialize()
    assert init.get("success") is True, f"Initialize failed: {init}"

    results = await svc.search_vectors(
        collection_name="c",
        query_vector=[0.0, 0.0, 0.0],
        limit=5,
    )
    assert isinstance(results, list) and results, "Empty or invalid results"
    first = results[0]
    assert set(first.keys()) == {"id", "score", "payload"}, f"Bad keys: {first.keys()}"

    res_ids = await svc.delete_points(collection_name="c", points_selector=models.PointIdsList(points=["a", "b"])) 
    assert res_ids.get("status") == "success", f"Delete by IDs failed: {res_ids}"

    res_filter = await svc.delete_points(collection_name="c", filter_conditions=models.FilterSelector(filter=models.Filter(must=[])))
    assert res_filter.get("status") == "success", f"Delete by filter failed: {res_filter}"

    print("OK: QdrantService contract validated locally")


if __name__ == "__main__":
    asyncio.run(main())
