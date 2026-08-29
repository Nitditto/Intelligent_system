"""Diabetes screening — Streamlit web client.

Thin UI over the FastAPI service (`api_client`). Four pages:
    1 Screening   2 Result   3 History   4 Dashboard (thresholds + monitoring)

Run:  streamlit run web/app.py      (with the API running on :8000)
"""
from __future__ import annotations

import datetime as dt
import uuid

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

import api_client as api

st.set_page_config(page_title="Diabetes Screening", page_icon="🩺", layout="wide")

# --------------------------------------------------------------- session bootstrap
if "session_id" not in st.session_state:
    st.session_state.session_id = "web-" + uuid.uuid4().hex[:8]
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_payload", None)


@st.cache_data(ttl=300, show_spinner=False)
def _questions():
    return api.questions()


@st.cache_data(ttl=300, show_spinner=False)
def _model_info():
    return api.model_info()


def _api_up() -> bool:
    try:
        api.health()
        return True
    except Exception:
        return False


# =============================================================== page: Screening
def page_screening():
    st.header("1 · Screening questionnaire")
    st.caption(
        "Answer what you can. Choose **Not sure** for anything you do not know — the "
        "estimate still works, with a note that it is less certain.")

    meta = _questions()
    questions = meta["questions"]
    sections: dict[str, list] = {}
    for q in questions:
        sections.setdefault(q["section"], []).append(q)

    answers: dict = {}
    with st.form("screening"):
        for section, qs in sections.items():
            st.subheader(section)
            cols = st.columns(2)
            for i, q in enumerate(qs):
                c = cols[i % 2]
                f = q["field"]
                label = q["label"]
                if q["type"] == "yesno":
                    v = c.radio(label, ["Yes", "No", "Not sure"], index=2,
                                horizontal=True, key=f)
                    answers[f] = {"Yes": 1, "No": 0, "Not sure": None}[v]
                elif q["type"] == "choice":
                    opts = q["options"]
                    labels = ["— Not sure —"] + [o[1] for o in opts]
                    idx = c.selectbox(label, range(len(labels)),
                                      format_func=lambda k, L=labels: L[k],
                                      index=0 if not q.get("required") else 1, key=f)
                    answers[f] = None if idx == 0 else opts[idx - 1][0]
                else:  # number
                    unknown = c.checkbox(f"{label}: not sure", value=False, key=f"{f}_na")
                    default = q.get("min", 0)
                    val = c.number_input(label, min_value=float(q.get("min", 0)),
                                         max_value=float(q.get("max", 300)),
                                         value=float(default), step=1.0,
                                         disabled=unknown, key=f)
                    answers[f] = None if unknown else float(val)
                if q.get("note"):
                    c.caption(q["note"])
        submitted = st.form_submit_button("Estimate diabetes risk", type="primary")

    if submitted:
        payload = {k: v for k, v in answers.items() if v is not None}
        payload["Age"] = answers.get("Age")
        payload["Sex"] = answers.get("Sex")
        payload["session_id"] = st.session_state.session_id
        if payload["Age"] is None or payload["Sex"] is None:
            st.error("Age group and Sex are required.")
            return
        try:
            with st.spinner("Scoring…"):
                res = api.predict(payload)
        except ValueError as e:
            st.error(f"Input rejected by the API: {e}")
            return
        except requests.RequestException as e:
            st.error(f"Could not reach the API: {e}")
            return
        st.session_state.last_result = res
        st.session_state.last_payload = payload
        st.success("Done — open **2 · Result** in the sidebar to see the full breakdown.")
        _result_summary(res)


def _result_summary(res: dict):
    band = res["band"]
    colour = {"Low": "success", "Moderate": "warning", "High": "error"}[band]
    pct = f"{res['probability'] * 100:.0f}%"
    c1, c2 = st.columns([1, 2])
    c1.metric("Estimated risk", pct, band)
    getattr(c2, colour)(res["band_label"])
    if res.get("uncertainty_band"):
        lo, hi = res["uncertainty_band"]
        c2.caption(f"Could range {lo*100:.0f}%–{hi*100:.0f}% depending on unanswered questions.")


# ================================================================= page: Result
def page_result():
    st.header("2 · Result")
    res = st.session_state.last_result
    if not res:
        st.info("Run a screening on page **1 · Screening** first.")
        return

    _result_summary(res)
    for w in res.get("warnings", []):
        st.warning(w)
    st.caption(
        f"Completeness {res['completeness']*100:.0f}% · "
        f"BMI source: {res['bmi_source']} · "
        "This is a screening aid, not a diagnosis.")
    st.divider()

    # --- SHAP factor bars ---
    st.subheader("Why this score")
    ex = res.get("explain") or {}
    if ex.get("factors"):
        df = pd.DataFrame(ex["factors"])[["label", "value"]].iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.barh(df["label"], df["value"],
                color=["#c44e52" if v > 0 else "#4c72b0" for v in df["value"]])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("contribution to risk  (→ increases,  ← decreases)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("SHAP values on the model's feature vector. Red raises the estimate, blue lowers it.")
    else:
        st.caption("Explanation not available (SHAP not installed on the API).")

    st.divider()

    # --- similar cases ---
    st.subheader("People similar to you in the survey data")
    sim = res.get("similar") or {}
    if sim.get("available"):
        st.metric("Closest matches who had diabetes / pre-diabetes",
                  f"{sim['n_with_diabetes']} of {sim['k']}")
        st.dataframe(pd.DataFrame(sim["neighbors"]), hide_index=True, width='stretch')
        st.caption(f"Method: {sim['method']}. Profiles are de-identified survey rows.")
    else:
        st.caption("Similar-case index not built on the API.")

    st.divider()

    # --- what-if / counterfactual ---
    st.subheader("What could change the estimate")
    wi = res.get("whatif") or {}
    factors = wi.get("factors", [])
    if factors:
        df = pd.DataFrame(factors)
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.barh(df["label"], df["delta"],
                color=["#4c72b0" if d < 0 else "#c44e52" for d in df["delta"]])
        ax.axvline(0, color="#333", lw=0.8)
        ax.set_xlabel("change in estimated risk if this factor were at its healthier value")
        fig.tight_layout()
        st.pyplot(fig)
    if wi.get("bmi_sweep"):
        sweep = pd.DataFrame(wi["bmi_sweep"]).set_index("bmi")
        st.line_chart(sweep, y="risk", height=240)
        st.caption(f"Estimated risk across BMI (your current BMI ≈ {wi.get('current_bmi')}).")

    cf = res.get("counterfactual") or {}
    if cf.get("changes"):
        parts = ", ".join(
            f"{c['label']} → {c['to']}" for c in cf["changes"])
        verdict = "would reach" if cf["feasible"] else "would still not reach"
        st.info(
            f"If **{parts}**, the estimate moves from {cf['from_risk']*100:.0f}% "
            f"({cf['from_band']}) to {cf['to_risk']*100:.0f}% ({cf['to_band']}) — "
            f"{verdict} the *{cf['target_band']}* band.")
    if wi.get("caveat"):
        st.caption("⚠️ " + wi["caveat"])


# ================================================================ page: History
def page_history():
    st.header("3 · History")
    sid = st.text_input("Session id", value=st.session_state.session_id)
    try:
        data = api.history(sid)
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return
    rows = data["screenings"]
    if not rows:
        st.info("No screenings recorded for this session yet.")
        return
    table = pd.DataFrame([{
        "time": dt.datetime.fromtimestamp(r["ts"]).strftime("%Y-%m-%d %H:%M"),
        "risk": f"{r['probability']*100:.0f}%",
        "band": r["band"],
        "completeness": f"{r['completeness']*100:.0f}%",
        "warnings": len(r["warnings"]),
    } for r in rows])
    st.dataframe(table, hide_index=True, width='stretch')


# ============================================================== page: Dashboard
def page_dashboard():
    st.header("4 · Operator dashboard")
    st.caption(
        "The decision cut-points are a clinic policy, not part of the model. Move the "
        "sliders to see the trade-off on held-out data, then save.")

    try:
        cfg = api.get_config()
        curve = api.threshold_curve("val")
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return
    if not curve.get("available"):
        st.warning("Holdout scores not built — run `python api/build_artifacts.py`.")
        return

    c1, c2 = st.columns(2)
    moderate_cut = c1.slider("moderate_cut", 0.0, 1.0, float(cfg["moderate_cut"]), 0.01)
    high_cut = c2.slider("high_cut", 0.0, 1.0, float(cfg["high_cut"]), 0.01)
    if moderate_cut > high_cut:
        st.error("moderate_cut must be ≤ high_cut")
        return

    pts = pd.DataFrame(curve["curve"])
    near = pts.iloc[(pts["high_cut"] - high_cut).abs().argmin()]
    m = st.columns(4)
    m[0].metric("Recall (sensitivity)", f"{near['recall']*100:.0f}%")
    m[1].metric("Precision", f"{near['precision']*100:.0f}%")
    m[2].metric("Flag rate", f"{near['flag_rate']*100:.0f}%")
    m[3].metric("F1", f"{near['f1']:.2f}")

    per_k = near["per_1000"]
    st.write(
        f"**Per 1,000 screened at this cut:** ~{per_k['flagged']} flagged · "
        f"~{per_k['caught']} true cases caught · ~{per_k['missed']} missed · "
        f"~{per_k['false_alarms']} false alarms.")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(pts["flag_rate"], pts["recall"], color="#4c72b0")
    ax.scatter([near["flag_rate"]], [near["recall"]], color="#c44e52", zorder=5, label="current")
    ax.set_xlabel("flag rate (share referred)")
    ax.set_ylabel("recall (share of cases caught)")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    # recommendation
    st.subheader("Recommend a cut-point")
    r1, r2, r3 = st.columns([1, 1, 1])
    obj = r1.selectbox("objective", ["recall", "flag_rate", "f1", "youden"])
    tgt = r2.number_input("target", 0.0, 1.0, 0.90, 0.05,
                          disabled=obj in ("f1", "youden"))
    if r3.button("Compute"):
        rec = api.threshold_recommend(obj, tgt)
        p = rec["point"]
        st.info(
            f"Suggested **high_cut = {p['high_cut']}** → recall {p['recall']*100:.0f}%, "
            f"precision {p['precision']*100:.0f}%, flag rate {p['flag_rate']*100:.0f}%.")

    # save
    st.subheader("Save policy")
    key = st.text_input("Admin key", type="password")
    if st.button("Save thresholds", type="primary"):
        try:
            out = api.set_thresholds(moderate_cut, high_cut, key)
            st.success(f"Saved: moderate_cut={out['high_cut']}… "
                       f"(now live for every screening).")
        except Exception as e:
            st.error(f"Rejected: {e}")

    st.divider()
    st.subheader("Live monitoring")
    mx = api.metrics()
    if mx.get("n_screenings", 0) == 0:
        st.info("No screenings logged yet.")
        return
    a = st.columns(4)
    a[0].metric("Screenings", mx["n_screenings"])
    a[1].metric("Flag rate (live)", f"{mx['flag_rate']*100:.0f}%")
    a[2].metric("BMI-clamp rate", f"{mx['bmi_capped_rate']*100:.0f}%")
    a[3].metric("Low-completeness rate", f"{mx['low_completeness_rate']*100:.0f}%")
    st.bar_chart(pd.Series(mx["band_distribution"]))
    if mx.get("mean_risk_by_age_band"):
        st.line_chart(pd.DataFrame(mx["mean_risk_by_age_band"]).set_index("age")["mean_probability"])


# ===================================================================== sidebar
PAGES = {
    "1 · Screening": page_screening,
    "2 · Result": page_result,
    "3 · History": page_history,
    "4 · Dashboard": page_dashboard,
}

with st.sidebar:
    st.title("🩺 Diabetes screening")
    if not _api_up():
        st.error(f"API not reachable at {api.API_URL}.\nStart it with "
                 "`uvicorn api.main:app --port 8000`.")
    else:
        mi = _model_info()
        st.caption(f"Model: {mi['chosen_model']} · seed {mi['random_seed']}")
        feats = mi["features_available"]
        st.caption("Features: " + ", ".join(k for k, v in feats.items() if v))
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.caption(f"session: `{st.session_state.session_id}`")

PAGES[choice]()
