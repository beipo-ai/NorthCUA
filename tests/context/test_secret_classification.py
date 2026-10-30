from __future__ import annotations

import pytest

from northcua.agentinfra.context.security import contains_sensitive_text, is_sensitive_key, redact_sensitive_text


@pytest.mark.parametrize(
    "key",
    [
        "X-API-Key",
        "access_token",
        "accessToken",
        "RefreshToken",
        "clientSecret",
        "sessionCookie",
        "Proxy-Authorization",
        "Set-Cookie",
        "awsSecretAccessKey",
        "secretAccessKey",
        "privateKey",
        "sshPrivateKey",
        "passwordHash",
        "cookieHeader",
        "authorizationHeader",
        "connectionString",
    ],
)
def test_sensitive_key_classifier_covers_common_naming_conventions(key: str) -> None:
    assert is_sensitive_key(key)


@pytest.mark.parametrize(
    "key",
    ["token_count", "token_counter", "cookie_policy", "authorization_status", "model_name"],
)
def test_sensitive_key_classifier_does_not_over_redact_safe_metadata(key: str) -> None:
    assert not is_sensitive_key(key)


@pytest.mark.parametrize(
    "text",
    [
        "secretAccessKey=abc",
        "privateKey: abc",
        "connectionString=postgres://user:pass@host/db",
        "postgres://user:pass@host/db",
    ],
)
def test_sensitive_text_classifier_rejects_labels_and_credential_uris(text: str) -> None:
    assert contains_sensitive_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "Ask the secretary to review",
        "authorization status is pending",
        "Use passwordless login",
    ],
)
def test_sensitive_text_classifier_allows_credential_related_natural_language(text: str) -> None:
    assert not contains_sensitive_text(text)


@pytest.mark.parametrize(
    ("text", "forbidden"),
    [
        ('{"awsSecretAccessKey":"abc123"}', "abc123"),
        ('cookieHeader="session=abc; preference=x y"', "preference=x y"),
        (
            'privateKey="-----BEGIN PRIVATE KEY-----\\nABCDEFSECRET\\n-----END PRIVATE KEY-----"',
            "ABCDEFSECRET",
        ),
    ],
)
def test_sensitive_text_redaction_consumes_complete_quoted_and_pem_values(text: str, forbidden: str) -> None:
    redacted = redact_sensitive_text(text)

    assert forbidden not in redacted
    assert "credential omitted" in redacted or "private key omitted" in redacted
