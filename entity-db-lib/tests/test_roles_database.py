"""Tests for RolesDatabase CRUD and search."""

import pytest

from corp_entity_db.store import RolesDatabase


def test_get_or_create_new_role(roles_db: RolesDatabase):
    """Creating a new role should return an ID > 0."""
    role_id = roles_db.get_or_create("Chief Executive Officer")
    assert role_id > 0


def test_get_or_create_idempotent(roles_db: RolesDatabase):
    """Calling get_or_create twice with the same name/source returns the same ID."""
    id1 = roles_db.get_or_create("Chief Financial Officer")
    id2 = roles_db.get_or_create("Chief Financial Officer")
    assert id1 == id2


def test_get_or_create_empty_name_raises(roles_db: RolesDatabase):
    """An empty role name should raise ValueError."""
    with pytest.raises(ValueError):
        roles_db.get_or_create("")


def test_search_exact_match(roles_db: RolesDatabase):
    """Searching for an exact role name should return score 1.0."""
    roles_db.get_or_create("Chief Executive Officer")
    results = roles_db.search("Chief Executive Officer")
    assert len(results) >= 1
    _id, name, score = results[0]
    assert name == "Chief Executive Officer"
    assert score == 1.0


def test_search_partial_match(roles_db: RolesDatabase):
    """A partial query like 'executive' should find 'Chief Executive Officer'."""
    roles_db.get_or_create("Chief Executive Officer")
    results = roles_db.search("executive")
    assert len(results) >= 1
    names = [name for _, name, _ in results]
    assert "Chief Executive Officer" in names


def test_search_no_match(roles_db: RolesDatabase):
    """Searching for a non-existent role should return an empty list."""
    results = roles_db.search("zzz_nonexistent_role_zzz")
    assert results == []


def test_get_by_id(roles_db: RolesDatabase):
    """get_by_id should return a RoleRecord with the correct name."""
    role_id = roles_db.get_or_create("General Counsel")
    record = roles_db.get_by_id(role_id)
    assert record is not None
    assert record.name == "General Counsel"


def test_get_stats(roles_db: RolesDatabase):
    """After inserting 3 roles, stats should show total_roles=3."""
    roles_db.get_or_create("CEO")
    roles_db.get_or_create("CFO")
    roles_db.get_or_create("CTO")
    stats = roles_db.get_stats()
    assert stats["total_roles"] == 3


def test_get_or_create_with_qid(roles_db: RolesDatabase):
    """Creating a role with a qid should store it correctly."""
    role_id = roles_db.get_or_create("Chief Executive Officer", qid=484876)
    record = roles_db.get_by_id(role_id)
    assert record is not None
    assert record.qid == 484876
