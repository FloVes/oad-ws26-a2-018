"""Product schema and deterministic Amazon metadata cleaning (no model calls)."""

import hashlib
import html
import math
import re
from html.parser import HTMLParser
from typing import Any

from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict, Field


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"br", "p", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parser = _TextExtractor()
    parser.feed(html.unescape(value))
    parser.close()  # Flush text buffered after an unterminated entity such as "&Blue".
    text = re.sub(r"\s+", " ", "".join(parser.parts)).strip()
    return "" if text.casefold() in {"none", "null", "n/a"} else text


def clean_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    # Preserve source order and remove repeated attributes without inventing data.
    return list(dict.fromkeys(text for part in value if (text := clean_text(part))))


def number(value: Any, low: float = 0, high: float = math.inf) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return result if math.isfinite(result) and low <= result <= high else None


def category_key(category: str) -> str:
    """Flat, collision-resistant Chroma key for an exact category label."""
    return "category_" + hashlib.sha256(category.strip().casefold().encode()).hexdigest()


class Product(BaseModel):
    """Authoritative product payload shared with later project phases."""

    model_config = ConfigDict(allow_inf_nan=False)
    item_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    features: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    price: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    average_rating: float | None = Field(default=None, ge=1, le=5)
    rating_number: int | None = Field(default=None, ge=0)
    brand: str | None = None
    image_url: str | None = None
    review_snippets: list[str] = Field(default_factory=list)
    source: str = "Amazon Reviews 2023 / Video_Games"

    def search_text(self) -> str:
        fields = [("Title", self.title), ("Categories", " | ".join(self.categories)),
                  ("Brand", self.brand), ("Features", "\n".join(self.features)),
                  ("Description", self.description)]
        return "\n".join(f"{label}: {value}" for label, value in fields if value)

    def as_document(self, text: str | None = None) -> Document:
        return Document(page_content=self.search_text() if text is None else text,
                        metadata=self.model_dump())

    def index_metadata(self) -> dict:
        metadata = {"item_id": self.item_id, "title": self.title}
        for name in ("price", "average_rating"):
            if (value := getattr(self, name)) is not None:
                metadata[name] = value
        for category in self.categories:
            metadata[category_key(category)] = True
        return metadata


def normalize_product(record: dict) -> Product | None:
    """Reject missing identity/title; retain products with optional data missing."""
    item_id = record.get("parent_asin")
    title = clean_text(record.get("title"))
    if not isinstance(item_id, str) or not item_id.strip() or not title:
        return None
    details = record.get("details")
    if not isinstance(details, dict):
        details = {}
    rating_number = number(record.get("rating_number"))
    images = record.get("images")
    image_url = None
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict):
                for key in ("hi_res", "large", "thumb"):
                    candidate = image.get(key)
                    if isinstance(candidate, str) and candidate.startswith(("https://", "http://")):
                        image_url = candidate
                        break
            if image_url:
                break
    return Product(
        item_id=item_id.strip(), title=title,
        description="\n".join(clean_list(record.get("description"))),
        features=clean_list(record.get("features")), categories=clean_list(record.get("categories")),
        price=number(record.get("price")), average_rating=number(record.get("average_rating"), 1, 5),
        rating_number=int(rating_number) if rating_number is not None and rating_number.is_integer() else None,
        brand=clean_text(details.get("Brand")) or None, image_url=image_url,
    )
