"""Canonical data contract tests, independent of any model or network."""

import math

import pytest

from src.rag.catalog import clean_text, normalize_product


def test_html_and_hidden_code_are_removed_without_inventing_attributes():
    product = normalize_product({
        "parent_asin": "  SOURCE-42  ",
        "title": "<b>Wireless &amp; comfortable</b><script>steal()</script>",
        "description": ["<p>For PC.</p><style>.secret {display:none}</style>", "Line<br>two"],
        "features": ["<b>USB</b>", "USB", "  ", None],
        "categories": ["Video Games", "PC", "PC"],
        "details": {"Brand": "<i>Example</i>"},
    })

    assert product is not None
    assert product.item_id == "SOURCE-42"
    assert product.title == "Wireless & comfortable"
    assert product.description == "For PC.\nLine two"
    assert product.features == ["USB"]
    assert product.categories == ["Video Games", "PC"]
    assert product.brand == "Example"
    assert "steal" not in product.search_text()
    assert "secret" not in product.search_text()


def test_missing_optional_fields_remain_explicitly_unknown_in_public_document():
    product = normalize_product({"parent_asin": "REAL-ID", "title": "Controller"})

    assert product is not None
    document = product.as_document()
    assert document.metadata["item_id"] == "REAL-ID"
    assert document.metadata["price"] is None  # Unknown does not mean free.
    assert document.metadata["average_rating"] is None
    assert document.metadata["rating_number"] is None
    assert document.metadata["description"] == ""
    assert document.metadata["features"] == []
    assert document.metadata["categories"] == []
    assert document.metadata["review_snippets"] == []
    assert document.metadata["currency"] == "USD"
    assert "Title: Controller" in document.page_content


@pytest.mark.parametrize("record", [
    {"title": "A product"},
    {"parent_asin": None, "title": "A product"},
    {"parent_asin": " ", "title": "A product"},
    {"parent_asin": 123, "title": "A product"},
    {"parent_asin": "KNOWN"},
    {"parent_asin": "KNOWN", "title": "<br>  "},
    {"parent_asin": "KNOWN", "title": "None"},
])
def test_missing_identity_or_title_does_not_create_a_product(record):
    assert normalize_product(record) is None


@pytest.mark.parametrize("price", [None, "None", "n/a", True, -1, math.inf, math.nan])
def test_invalid_prices_never_become_known_zero_prices(price):
    product = normalize_product({"parent_asin": "P", "title": "Game", "price": price})
    assert product is not None
    assert product.price is None


def test_valid_zero_and_numeric_price_are_preserved():
    free = normalize_product({"parent_asin": "P1", "title": "Game", "price": 0})
    priced = normalize_product({"parent_asin": "P2", "title": "Game", "price": "19.95"})
    assert free.price == 0
    assert priced.price == 19.95


@pytest.mark.parametrize("rating", [True, 0, 5.1, "None", math.inf])
def test_invalid_rating_does_not_turn_into_an_invented_valid_rating(rating):
    product = normalize_product({"parent_asin": "P", "title": "Game", "average_rating": rating})
    assert product.average_rating is None


def test_fractional_rating_count_is_unknown_and_real_image_is_retained():
    product = normalize_product({
        "parent_asin": "P", "title": "Game", "rating_number": 1.5,
        "images": [{"hi_res": None, "large": "https://example.org/game.jpg"}],
    })
    assert product.rating_number is None
    assert product.image_url == "https://example.org/game.jpg"


def test_oversized_numeric_fields_are_treated_as_unknown_not_a_loader_crash():
    product = normalize_product({
        "parent_asin": "P", "title": "Game", "price": 10**400,
        "average_rating": 10**400, "rating_number": 10**400,
    })
    assert product is not None
    assert product.price is None
    assert product.average_rating is None
    assert product.rating_number is None


def test_literal_comparison_text_and_entities_remain_readable():
    assert clean_text("For players aged 5 < 10 &amp; accessories") == "For players aged 5 < 10 & accessories"


def test_unterminated_entity_in_a_real_product_title_does_not_drop_the_product():
    title = "SADES Mechanical Gaming Keyboard(Black&Gold)"
    product = normalize_product({"parent_asin": "REAL", "title": title})
    assert product is not None
    assert product.title == title
