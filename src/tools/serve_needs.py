"""
Serve Needs List Tool
- Fetch approved Serve needs from external API
- Map complex API response to agent-friendly structure
- Support pagination and filtering
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
from datetime import datetime

router = APIRouter()

# --------- Models ---------

class TimeSlot(BaseModel):
    """Time slot for a specific day"""
    day: str
    startTime: str  # HH:MM format
    endTime: str    # HH:MM format

class ServeNeedItem(BaseModel):
    """Single Serve need item in simplified format"""
    needId: str
    title: str
    purpose: str
    status: str
    schoolName: str
    district: str
    state: str
    days: List[str]
    startDate: str  # YYYY-MM-DD
    endDate: str    # YYYY-MM-DD
    timeSlots: List[TimeSlot]
    detailsUrl: Optional[str] = None

class ServeNeedsListRequest(BaseModel):
    """Request parameters for listing Serve needs"""
    page: int = Field(default=0, ge=0, description="Page number (0-indexed)")
    size: int = Field(default=10, ge=1, le=20, description="Page size (max 20)")
    status: str = Field(default="Approved", description="Filter by status")

class ServeNeedsListResponse(BaseModel):
    """Response with paginated Serve needs"""
    page: int
    size: int
    totalElements: int
    totalPages: int
    items: List[ServeNeedItem]

# --------- Constants ---------

SERVE_API_BASE_URL = "https://serve-v1.evean.net/api/v1/serve-need/need/"
HTTP_TIMEOUT = 10.0  # seconds
MAX_RETRIES = 1

# --------- Helper Functions ---------

def _parse_iso_date(iso_str: Optional[str]) -> str:
    """Parse ISO date string to YYYY-MM-DD format"""
    if not iso_str:
        return ""
    try:
        # Clean up the string
        date_str = str(iso_str).strip()
        
        # Try parsing ISO format with timezone
        if 'T' in date_str:
            date_part = date_str.split('T')[0]
        else:
            date_part = date_str
        
        # Remove timezone suffix if present
        for suffix in ['Z', '+00:00', '+0000']:
            if date_part.endswith(suffix):
                date_part = date_part[:-len(suffix)]
        
        # Try ISO format
        if len(date_part) >= 10:
            dt = datetime.fromisoformat(date_part[:10])
            return dt.strftime('%Y-%m-%d')
        
        # Fallback: try simple date parsing
        dt = datetime.strptime(date_part[:10], '%Y-%m-%d')
        return dt.strftime('%Y-%m-%d')
    except (ValueError, AttributeError, IndexError):
        return ""

def _parse_time_to_hhmm(time_str: Optional[str]) -> str:
    """Parse time string to HH:MM format (24h)"""
    if not time_str:
        return ""
    try:
        # Try ISO time format (HH:MM:SS or HH:MM)
        if 'T' in time_str:
            time_str = time_str.split('T')[1]
        if '+' in time_str:
            time_str = time_str.split('+')[0]
        if 'Z' in time_str:
            time_str = time_str.replace('Z', '')
        
        # Extract HH:MM
        parts = time_str.split(':')
        if len(parts) >= 2:
            hour = parts[0].zfill(2)
            minute = parts[1].zfill(2)
            return f"{hour}:{minute}"
    except (ValueError, AttributeError, IndexError):
        pass
    return ""

def _parse_days(days_data: Any) -> List[str]:
    """Parse days from various formats"""
    if not days_data:
        return []
    
    if isinstance(days_data, list):
        # Already a list
        return [str(d).upper() if isinstance(d, str) else str(d) for d in days_data]
    elif isinstance(days_data, str):
        # Comma-separated string
        return [d.strip().upper() for d in days_data.split(',') if d.strip()]
    else:
        return []

def _extract_time_slots(time_slots_data: List[Dict[str, Any]]) -> List[TimeSlot]:
    """Extract time slots from timeSlots array in API response"""
    time_slots = []
    
    if not time_slots_data or not isinstance(time_slots_data, list):
        return []
    
    for slot in time_slots_data:
        if not isinstance(slot, dict):
            continue
        
        # Extract day (already a string like "Monday")
        day = slot.get('day', '')
        if not day:
            continue
        
        # Extract and parse start/end times (ISO datetime format)
        start_time = slot.get('startTime') or slot.get('start_time')
        end_time = slot.get('endTime') or slot.get('end_time')
        
        start_hhmm = _parse_time_to_hhmm(start_time)
        end_hhmm = _parse_time_to_hhmm(end_time)
        
        # Only add if we have valid time
        if start_hhmm and end_hhmm:
            time_slots.append(TimeSlot(
                day=str(day).upper(),  # Normalize to uppercase
                startTime=start_hhmm,
                endTime=end_hhmm
            ))
    
    return time_slots

def _map_serve_response_item(item: Dict[str, Any]) -> ServeNeedItem:
    """Map Serve API response item to simplified format"""
    # Extract need data
    need = item.get('need', {})
    entity = item.get('entity', {})
    occurrence = item.get('occurrence', {}) or item.get('occurrences', [{}])[0] if isinstance(item.get('occurrences'), list) and item.get('occurrences') else {}
    
    # Extract time slots from item level (not occurrence)
    time_slots_data = item.get('timeSlots', []) or item.get('time_slots', [])
    time_slots = _extract_time_slots(time_slots_data)
    
    # Extract days - prefer from occurrence, but also extract unique days from time slots as fallback
    days_from_occurrence = _parse_days(
        occurrence.get('days') or 
        need.get('days') or 
        item.get('days')
    )
    
    # Also extract unique days from time slots (more accurate)
    days_from_slots = []
    if time_slots_data:
        for slot in time_slots_data:
            if isinstance(slot, dict):
                day = slot.get('day')
                if day:
                    day_upper = str(day).upper()
                    if day_upper not in days_from_slots:
                        days_from_slots.append(day_upper)
    
    # Use days from time slots if available, otherwise use occurrence days
    days = days_from_slots if days_from_slots else days_from_occurrence
    
    # Extract dates
    start_date = _parse_iso_date(
        occurrence.get('startDate') or 
        occurrence.get('start_date') or
        need.get('startDate') or
        item.get('startDate')
    )
    end_date = _parse_iso_date(
        occurrence.get('endDate') or 
        occurrence.get('end_date') or
        need.get('endDate') or
        item.get('endDate')
    )
    
    # Extract entity details
    school_name = entity.get('name') or entity.get('schoolName') or need.get('schoolName') or ""
    district = entity.get('district') or need.get('district') or ""
    state = entity.get('state') or need.get('state') or ""
    details_url = entity.get('website') or entity.get('detailsUrl') or None
    
    return ServeNeedItem(
        needId=str(item.get('id') or item.get('needId') or need.get('id') or ""),
        title=need.get('name') or need.get('title') or item.get('title') or "",
        purpose=need.get('needPurpose') or need.get('purpose') or "",
        status=item.get('status') or need.get('status') or "Approved",
        schoolName=school_name,
        district=district,
        state=state,
        days=days,
        startDate=start_date,
        endDate=end_date,
        timeSlots=time_slots,
        detailsUrl=details_url
    )

# --------- Main Endpoint ---------

@router.post("/serve.needs.list", response_model=ServeNeedsListResponse)
async def serve_needs_list(req: ServeNeedsListRequest) -> ServeNeedsListResponse:
    """
    Fetch approved Serve needs from external API.
    Maps complex API response to agent-friendly structure.
    """
    # Validate size
    if req.size > 20:
        raise HTTPException(
            status_code=422,
            detail="Size must be <= 20"
        )
    
    # Build query parameters
    params = {
        "page": req.page,
        "size": req.size,
        "status": req.status
    }
    
    # Make HTTP request with retry
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.get(SERVE_API_BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                break
        except httpx.TimeoutException as e:
            last_error = f"Request timeout: {str(e)}"
            if attempt < MAX_RETRIES:
                continue
            raise HTTPException(
                status_code=504,
                detail="Failed to fetch Serve needs: Request timeout"
            )
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {str(e)}"
            if attempt < MAX_RETRIES:
                continue
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Serve needs: HTTP {e.response.status_code}"
            )
        except Exception as e:
            last_error = f"Network error: {str(e)}"
            if attempt < MAX_RETRIES:
                continue
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch Serve needs: {str(e)}"
            )
    
    # Parse response
    try:
        # Handle different response structures
        if isinstance(data, dict):
            # Paginated response
            items_data = data.get('content', data.get('items', data.get('data', [])))
            total_elements = data.get('totalElements', data.get('total', len(items_data)))
            total_pages = data.get('totalPages', data.get('total_pages', (total_elements + req.size - 1) // req.size))
            page = data.get('number', data.get('page', req.page))
            size = data.get('size', len(items_data) if not isinstance(items_data, list) else req.size)
        elif isinstance(data, list):
            # Simple list response
            items_data = data
            total_elements = len(items_data)
            total_pages = 1
            page = req.page
            size = req.size
        else:
            raise ValueError("Unexpected response format")
        
        # Map items
        mapped_items = []
        for item in items_data:
            try:
                mapped_item = _map_serve_response_item(item)
                mapped_items.append(mapped_item)
            except Exception as e:
                # Skip items that fail to parse
                print(f"[serve.needs.list] Warning: Failed to map item: {e}")
                continue
        
        return ServeNeedsListResponse(
            page=page,
            size=size,
            totalElements=total_elements,
            totalPages=total_pages,
            items=mapped_items
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse Serve needs response: {str(e)}"
        )

