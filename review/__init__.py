"""The review workspace: working with a meeting after it has finished.

Everything here reads a stored meeting rather than a live session, so it works
just as well on a meeting from last month as on the one that stopped a minute
ago. Three pieces:

- `retrieval` finds the passages that actually bear on a question, because a
  five-hour transcript does not fit in a prompt and paying to send all of it for
  every question would be absurd.
- `digest` reads the whole transcript once, in chunks, and condenses it. Reports
  are written from the digest, so a report covers the entire meeting without any
  single call having to hold it all.
- `reports` and `qa` are the two things a user does with the above.
"""
