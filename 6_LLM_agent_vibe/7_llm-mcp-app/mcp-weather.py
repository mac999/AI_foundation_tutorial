from typing import Any
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP
import sys

mcp = FastMCP("weather")

# 네이버 날씨 정보 도구
@mcp.tool()
def get_naver_weather(region: str) -> str:
    """
    네이버 날씨에서 특정 지역 날씨 정보를 가져옴.

    Args:
        region: 조회할 지역명 (예: 서울, 부산)
    """
    try:
        search_url = f"https://search.naver.com/search.naver?query={region}+날씨"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(search_url, headers=headers, verify=False)
        soup = BeautifulSoup(response.text, "html.parser")

        temperature = soup.select_one(".temperature_text > strong")
        status = soup.select_one(".weather_main")

        if not temperature or not status:
            return f"[{region}]의 날씨 정보를 불러올 수 없습니다. 지역명을 확인해 주세요."

        temp_text = temperature.get_text(strip=True)
        status_text = status.get_text(strip=True)

        return f"{region}의 현재 날씨는 '{status_text}'이며, 기온은 {temp_text}입니다."

    except Exception as e:
        print(e)
        return f"[오류] 날씨 정보를 가져오는 중 문제가 발생했습니다: {str(e)}"

# get_naver_weather('서울')

if __name__ == "__main__":
    print("MCP 서버 시작", file=sys.stderr)
    mcp.run()
