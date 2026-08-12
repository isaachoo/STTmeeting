"""The pre-meeting brief.

Everything the user knows before the meeting starts, in one structured object.
It feeds three places: the prompts (so advice is specific rather than generic),
Deepgram's keyterm boosting (so jargon and names survive transcription), and the
saved session record.
"""

from dataclasses import dataclass, field


@dataclass
class Attendee:
    name: str
    role: str = ""
    is_me: bool = False

    def label(self) -> str:
        if self.role and self.is_me:
            return f"{self.name} ({self.role}) -- this is the user"
        if self.role:
            return f"{self.name} ({self.role})"
        if self.is_me:
            return f"{self.name} -- this is the user"
        return self.name

    def as_dict(self) -> dict:
        return {"name": self.name, "role": self.role, "is_me": self.is_me}


@dataclass
class Brief:
    title: str = ""
    context: str = ""
    agenda: str = ""
    my_goal: str = ""
    my_role: str = ""
    attendees: list[Attendee] = field(default_factory=list)
    glossary: list[str] = field(default_factory=list)

    # ------------------------------------------------------------ construction

    @classmethod
    def from_payload(cls, data: dict | None) -> "Brief":
        """Build from whatever the browser sent, defensively."""
        data = data or {}

        attendees = []
        for raw in _as_list(data.get("attendees"))[:12]:
            if isinstance(raw, dict):
                name = _text(raw.get("name"), 80)
                if not name:
                    continue
                attendees.append(
                    Attendee(
                        name=name,
                        role=_text(raw.get("role"), 120),
                        is_me=bool(raw.get("is_me")),
                    )
                )
            elif _text(raw, 80):
                attendees.append(Attendee(name=_text(raw, 80)))

        glossary = []
        for raw in _as_list(data.get("glossary"))[:60]:
            term = _text(raw, 60)
            if term and term not in glossary:
                glossary.append(term)

        return cls(
            title=_text(data.get("title"), 200),
            context=_text(data.get("context"), 6000),
            agenda=_text(data.get("agenda"), 4000),
            my_goal=_text(data.get("my_goal"), 2000),
            my_role=_text(data.get("my_role"), 500),
            attendees=attendees,
            glossary=glossary,
        )

    @classmethod
    def from_dict(cls, data: dict | None) -> "Brief":
        """Rebuild a stored brief (same shape as as_dict)."""
        return cls.from_payload(data)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "context": self.context,
            "agenda": self.agenda,
            "my_goal": self.my_goal,
            "my_role": self.my_role,
            "attendees": [a.as_dict() for a in self.attendees],
            "glossary": list(self.glossary),
        }

    # -------------------------------------------------------------- rendering

    @property
    def me(self) -> Attendee | None:
        for attendee in self.attendees:
            if attendee.is_me:
                return attendee
        return None

    def keyterms(self) -> list[str]:
        """Terms worth boosting in the transcriber: jargon plus every name."""
        terms = list(self.glossary)
        for attendee in self.attendees:
            if attendee.name and attendee.name not in terms:
                terms.append(attendee.name)
        return terms

    def is_empty(self) -> bool:
        return not any(
            (self.context, self.agenda, self.my_goal, self.my_role,
             self.attendees, self.glossary)
        )

    def role_line(self) -> str:
        """How the copilot should think of the user. Everything it suggests is
        judged from this seat, so an empty role must still say something."""
        if self.my_role:
            return self.my_role
        me = self.me
        if me and me.role:
            return me.role
        return "a general participant, with no particular stake stated"

    def render(self) -> str:
        """The brief as prompt text. Sections are omitted when empty."""
        parts: list[str] = []
        if self.title:
            parts.append(f"Meeting: {self.title}")
        if self.attendees:
            people = "\n".join(f"- {a.label()}" for a in self.attendees)
            parts.append(f"In the room:\n{people}")
        if self.agenda:
            parts.append(f"Agenda:\n{self.agenda}")
        parts.append(f"The user's role in this meeting:\n{self.role_line()}")
        if self.my_goal:
            parts.append(f"What the user wants out of this meeting:\n{self.my_goal}")
        if self.context:
            parts.append(f"Background:\n{self.context}")
        if self.glossary:
            parts.append(
                "Terms, names and jargon that will appear (the transcriber often "
                "mangles these -- correct them silently):\n"
                + ", ".join(self.glossary)
            )
        return "\n\n".join(parts)


def _text(value, limit: int) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()[:limit]


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        # Accept a comma or newline separated string for convenience.
        return [p for p in (x.strip() for x in value.replace("\n", ",").split(",")) if p]
    return []
