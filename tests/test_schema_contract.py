"""Schema contract tests for notice_schema_v1.

Verifies that:
- A valid v1 payload passes validation.
- An additive (backward-compatible) schema change still accepts existing payloads.
- Breaking changes (missing required field, type mismatch, version bump) are rejected.
"""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft7Validator, ValidationError, validate

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "notice_schema_v1.json"


@pytest.fixture
def schema_v1() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def valid_payload() -> dict:
    return {
        "schema_version": 1,
        "notice_id": "DN-001",
        "domain_name": "example-cloud.net",
        "registrar": "Namecheap",
        "notice_type": "expiration_warning",
        "notice_date": "2026-03-15",
        "action_required": True,
        "deadline": "2026-04-15",
        "amount_due_usd": 14.99,
    }


# ── Baseline ──────────────────────────────────────────────────────────

class TestBaselineValidation:
    def test_valid_v1_payload_passes(self, schema_v1, valid_payload):
        """A correct v1 payload must validate without errors."""
        validate(instance=valid_payload, schema=schema_v1)

    def test_valid_payload_with_null_optionals(self, schema_v1, valid_payload):
        """Nullable fields accept null values."""
        payload = {**valid_payload, "deadline": None, "amount_due_usd": None}
        validate(instance=payload, schema=schema_v1)

    def test_all_notice_types_accepted(self, schema_v1, valid_payload):
        """Every enum value for notice_type must be accepted."""
        for ntype in [
            "expiration_warning", "renewal_notice", "transfer_request",
            "suspension_alert", "whois_verification", "dnssec_alert",
        ]:
            payload = {**valid_payload, "notice_type": ntype}
            validate(instance=payload, schema=schema_v1)


# ── Additive Changes (must pass) ─────────────────────────────────────

class TestAdditiveChanges:
    def test_additive_optional_field_accepted(self, schema_v1, valid_payload):
        """Adding an optional field to the schema is backward-compatible:
        existing payloads still validate, and payloads with the new field
        also validate under the evolved schema."""
        evolved_schema = copy.deepcopy(schema_v1)
        evolved_schema["properties"]["urgent_flag"] = {"type": "boolean"}
        # Remove additionalProperties restriction for additive test
        evolved_schema.pop("additionalProperties", None)

        # Existing payload (without new field) still passes
        validate(instance=valid_payload, schema=evolved_schema)

        # Payload with new optional field also passes
        payload_with_new = {**valid_payload, "urgent_flag": True}
        validate(instance=payload_with_new, schema=evolved_schema)

    def test_additive_notes_field_accepted(self, schema_v1, valid_payload):
        """Adding an optional 'notes' string field is backward-compatible."""
        evolved_schema = copy.deepcopy(schema_v1)
        evolved_schema["properties"]["notes"] = {"type": "string"}
        evolved_schema.pop("additionalProperties", None)

        validate(instance=valid_payload, schema=evolved_schema)

        payload_with_notes = {**valid_payload, "notes": "Follow up with registrar."}
        validate(instance=payload_with_notes, schema=evolved_schema)


# ── Breaking Changes (must fail) ─────────────────────────────────────

class TestBreakingChanges:
    def test_fails_missing_required_domain_name(self, schema_v1, valid_payload):
        """Removing a required field (domain_name) must fail validation."""
        broken = {k: v for k, v in valid_payload.items() if k != "domain_name"}
        with pytest.raises(ValidationError) as exc_info:
            validate(instance=broken, schema=schema_v1)
        assert "domain_name" in exc_info.value.message

    def test_fails_missing_required_notice_type(self, schema_v1, valid_payload):
        """Removing a required field (notice_type) must fail validation."""
        broken = {k: v for k, v in valid_payload.items() if k != "notice_type"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_missing_required_notice_date(self, schema_v1, valid_payload):
        """Removing a required field (notice_date) must fail validation."""
        broken = {k: v for k, v in valid_payload.items() if k != "notice_date"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_type_mismatch_action_required_string(self, schema_v1, valid_payload):
        """Changing action_required from boolean to string must fail."""
        broken = {**valid_payload, "action_required": "yes"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_type_mismatch_amount_due_string(self, schema_v1, valid_payload):
        """Changing amount_due_usd from number to non-null string must fail."""
        broken = {**valid_payload, "amount_due_usd": "14.99"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_invalid_notice_type_enum(self, schema_v1, valid_payload):
        """Using a notice_type value outside the enum must fail."""
        broken = {**valid_payload, "notice_type": "domain_purchase"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_schema_version_mismatch(self, schema_v1, valid_payload):
        """A schema_version != 1 must fail the const constraint."""
        broken = {**valid_payload, "schema_version": 2}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_schema_version_string(self, schema_v1, valid_payload):
        """schema_version as string must fail type constraint."""
        broken = {**valid_payload, "schema_version": "1"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)

    def test_fails_additional_property_strict(self, schema_v1, valid_payload):
        """With additionalProperties: false, extra fields must fail."""
        broken = {**valid_payload, "extra_field": "should_not_be_here"}
        with pytest.raises(ValidationError):
            validate(instance=broken, schema=schema_v1)
