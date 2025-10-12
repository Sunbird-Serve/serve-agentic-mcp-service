#!/usr/bin/env python3
"""
Test script to debug time parsing issues
"""
import asyncio
import httpx
from datetime import datetime
import pytz

async def test_time_parsing():
    """Test time parsing for specific cases"""
    
    url = "http://localhost:9000/mcp/time.parse_options"
    
    test_cases = [
        "tomorrow 3 PM",
        "sunday 4 PM",
        "tomorrow 3 PM or sunday 4 PM",
        "next monday 5 PM",
        "saturday 2 PM",
    ]
    
    print("="*60)
    print("TIME PARSING TEST")
    print("="*60)
    print(f"\nCurrent time: {datetime.now(pytz.timezone('Asia/Kolkata'))}")
    print()
    
    async with httpx.AsyncClient() as client:
        for test_input in test_cases:
            print(f"\n📝 Input: '{test_input}'")
            print("-" * 60)
            
            try:
                response = await client.post(
                    url,
                    json={
                        "text": test_input,
                        "tz": "Asia/Kolkata",
                        "duration_minutes": 30
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("slots"):
                        print(f"✅ Parsed {len(data['slots'])} slot(s):")
                        for i, slot in enumerate(data['slots'], 1):
                            print(f"   {i}. {slot['label']}")
                            print(f"      Start: {slot['start_iso']}")
                    else:
                        print(f"⚠️  No slots parsed")
                        if data.get("needs_clarification"):
                            print(f"      Reason: {data.get('reason')}")
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"   {response.text}")
            
            except Exception as e:
                print(f"❌ Exception: {e}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_time_parsing())

