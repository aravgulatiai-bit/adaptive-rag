# InsightHub API Authentication Guide

The InsightHub REST API lets developers programmatically manage data sources,
dashboards, and alerts. This guide explains how to authenticate.

## Authentication methods

InsightHub supports two authentication methods:

1. **OAuth 2.0** for user-facing integrations, using the authorization code
   flow.
2. **API keys** for server-to-server automation.

## Tokens

OAuth access tokens expire after 1 hour. A refresh token, valid for 30 days, can
be used to obtain a new access token without the user signing in again. If a
refresh token is unused for 30 days it is revoked automatically.

## API keys

API keys are created in the InsightHub admin console under Settings then API
Keys. A key is shown only once at creation time and must be stored securely. Keys
can be scoped to specific permissions (read-only or read-write) and can be
revoked at any time.

## Rate limits

The API allows 600 requests per minute per API key. Requests beyond the limit
receive an HTTP 429 response with a Retry-After header. Enterprise customers can
request a higher limit.

## Security

All API traffic must use HTTPS. Requests over plain HTTP are rejected. Never
embed API keys in client-side code. For help, contact
developers@northwind.example.
