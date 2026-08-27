import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import re
import base64

def normalize_vietnamese_text(text: str) -> str:
    if not text:
        return ""
    return text.strip().lower()

def clean_location(city: str, district: str):
    clean_c = city.replace("Thành phố ", "").replace("Tỉnh ", "").strip()
    clean_d = district.replace("Quận ", "").replace("Huyện ", "").replace("Thị xã ", "").strip()
    return clean_c, clean_d

def build_queries(city, district, price, area=None, bedrooms=None):
    clean_c, clean_d = clean_location(city, district)
    
    # Round price to 1 decimal place to match real listings
    rounded_price = round(price, 1)
    if rounded_price.is_integer():
        price_str = str(int(rounded_price))
    else:
        price_str = f"{rounded_price:.1f}"
        
    area_str = f"{int(area)}" if area else None
    
    queries = []
    
    # 1. Targeted query with price
    queries.append(f"bán nhà {clean_d} {clean_c} {price_str} tỷ")
    
    # 2. Fallback query without price to guarantee results
    queries.append(f"bán nhà {clean_d} {clean_c}")
    
    # 3. Targeted query with area and price
    if area_str:
        queries.append(f"bán nhà {clean_d} {clean_c} {area_str}m2 {price_str} tỷ")
        
    # 4. Fallback query with area
    if area_str:
        queries.append(f"bán nhà {clean_d} {clean_c} {area_str}m2")
        
    # Site specific queries
    for site in ['batdongsan.com.vn', 'alonhadat.com.vn', 'muaban.net']:
        queries.append(f"site:{site} bán nhà {clean_d} {clean_c} {price_str} tỷ")
        
    return queries

def parse_price(text: str):
    import re
    text = text.lower()
    # 5 tỷ 800 triệu
    match2 = re.search(r'([0-9]+)\s*tỷ\s*([0-9]+)\s*triệu', text)
    if match2:
        try:
            return float(match2.group(1)) + float(match2.group(2)) / 1000.0
        except:
            pass
    # 6 tỷ, 6.5 tỷ, 6,5 tỷ
    match = re.search(r'([0-9]+[.,]?[0-9]*)\s*(tỷ|t\.)', text)
    if match:
        val = match.group(1).replace(',', '.')
        try:
            return float(val)
        except:
            pass
    # 5800 triệu
    match3 = re.search(r'([0-9]{4,})\s*triệu', text)
    if match3:
        try:
            return float(match3.group(1)) / 1000.0
        except:
            pass
    return None

def parse_area(text: str):
    import re
    match = re.search(r'(?:diện tích|dt|dtich)?\s*([0-9]+[.,]?[0-9]*)\s*(?:m2|m²|m\b|mét)', text.lower())
    if match:
        val = match.group(1).replace(',', '.')
        try:
            return float(val)
        except:
            pass
    return None

def score_result(result, district, city, predicted_price, target_area=None):
    score = 0
    title = result.get('title', '').lower()
    snippet = result.get('body', '').lower()
    href = result.get('href', '').lower()
    text_to_search = title + " " + snippet
    
    clean_c, clean_d = clean_location(city, district)
    
    if clean_d.lower() in text_to_search:
        score += 40
    if clean_c.lower() in text_to_search:
        score += 25
        
    found_price = parse_price(text_to_search)
    if found_price:
        result['matched_price'] = found_price
        # +20 nếu giá nằm trong khoảng predicted_price ± 20%
        if predicted_price * 0.8 <= found_price <= predicted_price * 1.2:
            score += 20
            
    found_area = parse_area(text_to_search)
    if found_area:
        result['matched_area'] = found_area
        if target_area and target_area * 0.75 <= found_area <= target_area * 1.25:
            score += 10
            
    bds_sites = ['batdongsan.com.vn', 'alonhadat.com.vn', 'muaban.net', 'bds68.com.vn', '123nhadatviet.net', 'nhadat24h.net']
    if any(site in href for site in bds_sites):
        score += 10
        
    bad_sites = ['bing.com', 'chotot.com', 'nhatot.com', 'shopee', 'facebook.com', 'youtube.com']
    if any(site in href for site in bad_sites):
        score -= 50
    if 'chợ tốt' in text_to_search or 'nền tảng' in text_to_search or 'rao vặt' in text_to_search:
        score -= 50
        
    result['score'] = score
    return score

def search_ddg(query: str, max_results=10):
    results = []
    
    # 1. Try html.duckduckgo.com POST method first (extremely robust, bypasses bot blocks)
    try:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({'q': query}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        html = urllib.request.urlopen(req, timeout=8).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        for div in soup.select('.result'):
            title_elem = div.select_one('.result__a')
            snippet_elem = div.select_one('.result__snippet')
            
            if title_elem:
                href = title_elem.get('href', '')
                if 'uddg=' in href:
                    m = re.search(r'uddg=([^&]+)', href)
                    if m:
                        href = urllib.parse.unquote(m.group(1))
                
                results.append({
                    'title': title_elem.get_text(strip=True),
                    'href': href,
                    'body': snippet_elem.get_text(strip=True) if snippet_elem else "Đang cập nhật...",
                    'source': 'DuckDuckGo'
                })
                if len(results) >= max_results:
                    break
        if results:
            print(f"DuckDuckGo HTML parsed {len(results)} results.")
            return results
    except Exception as e:
        print("DuckDuckGo HTML scraper error:", e)

    # 2. Fallback to DDGS library if POST scraper fails
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, region='vn-vi', max_results=max_results):
                results.append({
                    'title': r.get('title', ''),
                    'href': urllib.parse.unquote(r.get('href', '')),
                    'body': r.get('body', ''),
                    'source': 'DuckDuckGo'
                })
    except Exception as e:
        print("DDGS library error:", e)
    return results

def search_bing(query: str):
    results = []
    try:
        bing_url = f'https://www.bing.com/search?q={urllib.parse.quote(query)}'
        req = urllib.request.Request(bing_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        html = urllib.request.urlopen(req).read()
        soup = BeautifulSoup(html, 'html.parser')
        for li in soup.select('li.b_algo'):
            h2_a = li.select_one('h2 a')
            if h2_a:
                snippet_elem = li.select_one('.b_caption p')
                href = urllib.parse.unquote(h2_a['href'])
                if 'u=' in href:
                    m = re.search(r'u=a1([a-zA-Z0-9]+)', href)
                    if m:
                        try:
                            href = base64.b64decode(m.group(1)).decode('utf-8')
                        except:
                            pass
                results.append({
                    'title': h2_a.get_text(),
                    'href': href,
                    'body': snippet_elem.get_text() if snippet_elem else "Đang cập nhật...",
                    'source': 'Bing'
                })
    except Exception as e:
        print("Bing error:", e)
    return results

def search_google(query: str, max_results=10):
    results = []
    try:
        from googlesearch import search as google_search
        for url in google_search(query, num_results=max_results, advanced=True):
            results.append({
                'title': url.title,
                'href': urllib.parse.unquote(url.url),
                'body': url.description,
                'source': 'Google'
            })
    except Exception as e:
        pass
    return results

def dedupe_results(results):
    seen = set()
    unique = []
    for r in results:
        href = r.get('href', '').lower()
        if href not in seen:
            seen.add(href)
            unique.append(r)
    return unique

def search_images_ddg(query: str, max_results=3):
    images = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.images(query, region='vn-vi', max_results=max_results):
                images.append(r.get('image', ''))
    except Exception:
        pass
    return images
