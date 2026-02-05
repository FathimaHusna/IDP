from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import Optional, List


class LineItem(BaseModel):
    description: Optional[str]
    quantity: Optional[float] = Field(ge=0.0, default=None)
    unit_price: Optional[float] = Field(ge=0.0, default=None)
    line_total: Optional[float] = Field(ge=0.0, default=None)


class InvoicePayload(BaseModel):
    vendor: Optional[str]
    invoice_number: Optional[str]
    subtotal: Optional[float]
    tax: Optional[float]
    total: Optional[float]
    lpo_ref: Optional[str]
    payment_received: Optional[float]
    currency: Optional[str]
    line_items: Optional[List[LineItem]]

    @validator("subtotal", "tax", "total", "payment_received")
    def non_negative(cls, v):
        if v is None:
            return v
        if float(v) < 0:
            raise ValueError("negative amount")
        return v


class POPayload(BaseModel):
    vendor: Optional[str]
    po_number: Optional[str]
    lpo_ref: Optional[str]
    total_amount: Optional[float]
    currency: Optional[str]
    line_items: Optional[List[LineItem]]

    @validator("total_amount")
    def non_negative_total(cls, v):
        if v is None:
            return v
        if float(v) < 0:
            raise ValueError("negative amount")
        return v


class ReceiptPayload(BaseModel):
    vendor: Optional[str]
    reference: Optional[str]
    lpo_ref: Optional[str]
    invoice_ref: Optional[str]
    amount: Optional[float]
    currency: Optional[str]

    @validator("amount")
    def non_negative_amount(cls, v):
        if v is None:
            return v
        if float(v) < 0:
            raise ValueError("negative amount")
        return v


def validate_payload_schema(doc_type: str, payload: dict) -> list[str]:
    try:
        if doc_type == "invoice":
            InvoicePayload(**payload)
        elif doc_type == "po":
            POPayload(**payload)
        elif doc_type == "receipt":
            ReceiptPayload(**payload)
        else:
            return []
        return []
    except Exception as e:
        return [str(e)]

