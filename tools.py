"""
사내 데이터 조회 도구.

inventory-dashboard 프로젝트가 사용하는 JSON 파일을 그대로 읽습니다.
각 함수는 LLM이 function calling 으로 호출하며, 반환값은 JSON 직렬화 가능한 dict 입니다.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _load(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _norm(s):
    """품목명 비교용 정규화 — 공백과 대소문자 차이를 무시한다."""
    return "".join(str(s).split()).lower()


# ---------------------------------------------------------------- 재고 조회

def get_inventory(keyword: str, warehouse: str = "") -> dict:
    """품번 또는 품목명으로 재고를 조회하고 안전재고와 비교한다."""
    rows = _load("demo_inventory.json")
    kpi = _load("kpi_settings.json")

    # (품번, 창고) -> 안전재고
    safe_map = {}
    for k in kpi:
        safe_map[(k.get("code", ""), k.get("warehouse", ""))] = k.get("safe_qty", 0)

    kw = _norm(keyword)
    hits = []
    for r in rows:
        if kw and kw not in _norm(r["PROD_CD"]) and kw not in _norm(r["PROD_DES"]):
            continue
        if warehouse and _norm(warehouse) not in _norm(r["WH_DES"]):
            continue

        qty = int(r["BAL_QTY"])
        safe = safe_map.get((r["PROD_CD"], r["WH_DES"]))
        item = {
            "품번": r["PROD_CD"],
            "품목명": r["PROD_DES"],
            "창고": r["WH_DES"],
            "재고수량": qty,
        }
        if safe is not None:
            item["안전재고"] = safe
            item["안전재고_미달"] = qty < safe
        hits.append(item)

    if not hits:
        return {
            "결과": [],
            "안내": f"'{keyword}' 와 일치하는 품목이 재고 데이터에 없습니다. "
                    "품목명 일부만으로 다시 조회해 보세요.",
        }
    return {"결과": hits, "건수": len(hits)}


# ---------------------------------------------------------------- BOM 전개

def _build_children_map(bom):
    children = {}
    for node in bom:
        children.setdefault(node.get("parentId"), []).append(node)
    return children


def _find_root(bom, keyword):
    kw = _norm(keyword)
    roots = [n for n in bom if n.get("parentId") is None]
    for n in roots:
        if kw in _norm(n["code"]) or kw in _norm(n["name"]):
            return n
    # 최상위에 없으면 중간 노드까지 검색
    for n in bom:
        if kw in _norm(n["code"]) or kw in _norm(n["name"]):
            return n
    return None


def get_bom(keyword: str, production_qty: int = 1) -> dict:
    """완제품 BOM을 전개하고 생산수량에 따른 부속품 소요량을 계산한다."""
    bom = _load("bom_data.json")
    root = _find_root(bom, keyword)
    if root is None:
        available = sorted({n["name"] for n in bom if n.get("parentId") is None})
        return {
            "안내": f"'{keyword}' 에 해당하는 BOM이 없습니다.",
            "등록된_완제품": available,
        }

    children = _build_children_map(bom)
    flat = []

    def walk(node, multiplier, depth):
        req = node.get("reqQty", 1) or 1
        total = multiplier * req
        if depth > 0:  # 최상위 완제품 자신은 소요량 목록에서 제외
            flat.append({
                "레벨": depth,
                "품번": node["code"],
                "품목명": node["name"],
                "단위소요량": req,
                "총소요량": total,
                "공급업체": node.get("supplier", ""),
            })
        for c in children.get(node["id"], []):
            walk(c, total, depth + 1)

    walk(root, production_qty, 0)

    return {
        "완제품": {"품번": root["code"], "품목명": root["name"]},
        "생산수량": production_qty,
        "소요_부속품": flat,
    }


# ---------------------------------------------------------- 생산 가능성 판정

def check_production_feasibility(product: str, production_qty: int) -> dict:
    """BOM 소요량과 현재 재고를 대조해 생산 가능 여부와 부족분을 판정한다."""
    bom_result = get_bom(product, production_qty)
    if "소요_부속품" not in bom_result:
        return bom_result

    rows = _load("demo_inventory.json")
    stock = {}
    for r in rows:
        stock[r["PROD_CD"]] = stock.get(r["PROD_CD"], 0) + int(r["BAL_QTY"])

    suppliers = {s["name"]: s for s in _load("suppliers.json")}

    shortages, ok = [], []
    for part in bom_result["소요_부속품"]:
        have = stock.get(part["품번"], 0)
        need = part["총소요량"]
        entry = {
            "품번": part["품번"],
            "품목명": part["품목명"],
            "필요수량": need,
            "보유재고": have,
        }
        if have < need:
            entry["부족수량"] = need - have
            sup = suppliers.get(part["공급업체"])
            if sup:
                entry["공급업체"] = sup["name"]
                entry["리드타임"] = sup.get("leadTime", "")
                entry["담당자"] = sup.get("manager", "")
            else:
                entry["공급업체"] = part["공급업체"] or "미등록"
            shortages.append(entry)
        else:
            ok.append(entry)

    return {
        "완제품": bom_result["완제품"],
        "생산수량": production_qty,
        "생산가능": len(shortages) == 0,
        "부족_품목": shortages,
        "충분_품목": ok,
        "주의": "재고 데이터에 등록되지 않은 부속품은 보유재고 0으로 계산됩니다.",
    }


# ---------------------------------------------------------------- 협력업체

def get_supplier(keyword: str = "") -> dict:
    """협력업체 연락처와 리드타임을 조회한다."""
    suppliers = _load("suppliers.json")
    if not keyword:
        return {"결과": suppliers}

    kw = _norm(keyword)
    hits = [
        s for s in suppliers
        if kw in _norm(s["name"]) or kw in _norm(s.get("memo", ""))
    ]
    if not hits:
        return {"결과": [], "안내": f"'{keyword}' 와 일치하는 협력업체가 없습니다."}
    return {"결과": hits}


# -------------------------------------------------- function calling 스키마

TOOL_SCHEMAS = [
    {
        "name": "get_inventory",
        "description": (
            "품번 또는 품목명으로 현재 재고 수량을 창고별로 조회한다. "
            "안전재고가 설정된 품목은 미달 여부도 함께 반환한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "품번(예: FP-105) 또는 품목명 일부(예: S26 밴드)",
                },
                "warehouse": {
                    "type": "string",
                    "description": "창고명으로 좁힐 때만 사용 (예: 본사창고, 공장창고)",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "get_bom",
        "description": (
            "완제품의 BOM 계층 구조를 전개하고, 생산수량에 따른 "
            "부속품별 총소요량을 계산한다. 재고와 대조하지는 않는다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "완제품 품번 또는 품목명"},
                "production_qty": {
                    "type": "integer",
                    "description": "생산 예정 수량. 지정이 없으면 1",
                },
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "check_production_feasibility",
        "description": (
            "특정 완제품을 지정 수량만큼 생산할 수 있는지 판정한다. "
            "BOM 소요량과 현재 재고를 대조해 부족한 부속품과 "
            "해당 공급업체 리드타임까지 함께 반환한다. "
            "'N대 생산 가능한가', '부속품이 충분한가' 류의 질문에 사용한다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "product": {"type": "string", "description": "완제품 품번 또는 품목명"},
                "production_qty": {"type": "integer", "description": "생산 예정 수량"},
            },
            "required": ["product", "production_qty"],
        },
    },
    {
        "name": "get_supplier",
        "description": "협력업체의 담당자, 연락처, 평균 리드타임을 조회한다.",
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "업체명 또는 취급 품목. 비우면 전체 목록",
                }
            },
        },
    },
]

TOOL_FUNCS = {
    "get_inventory": get_inventory,
    "get_bom": get_bom,
    "check_production_feasibility": check_production_feasibility,
    "get_supplier": get_supplier,
}
