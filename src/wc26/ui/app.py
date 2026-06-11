"""Thin read-only Flask app over the prediction artifact (analyst UI).

No auth, no writes, no background jobs — it renders the precomputed artifact
and the existing reports. Run via `wc26 serve`.
"""
from __future__ import annotations

from flask import Flask, abort, render_template, request

from wc26.ui import data as uidata


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    @app.route("/")
    def matches():
        art = uidata.load_artifact()
        rows = art["matches"]
        f_status = request.args.get("status", "")
        f_stage = request.args.get("stage", "")
        f_team = request.args.get("team", "").strip()
        f_conf = request.args.get("confidence", "")

        def keep(m):
            if f_status and m["status"] != f_status:
                return False
            if f_stage == "group" and m.get("is_knockout"):
                return False
            if f_stage == "knockout" and not m.get("is_knockout"):
                return False
            if f_team and f_team.lower() not in f"{m['home']} {m['away']}".lower():
                return False
            if f_conf and m.get("confidence") != f_conf:
                return False
            return True

        shown = [m for m in rows if keep(m)]
        teams = sorted({m["home"] for m in rows if not m.get("is_knockout")}
                       | {m["away"] for m in rows if not m.get("is_knockout")})
        return render_template(
            "matches.html", artifact=art, matches=shown, teams=teams,
            filters={"status": f_status, "stage": f_stage, "team": f_team, "confidence": f_conf},
        )

    @app.route("/match/<match_id>")
    def match(match_id):
        art = uidata.load_artifact()
        m = uidata.match_by_id(art, match_id)
        if m is None:
            abort(404)
        history = uidata.match_history(m["home"], m["away"]) if m["status"] != "pending" else []
        return render_template("match.html", artifact=art, m=m, history=history)

    @app.route("/evaluation")
    def evaluation():
        return render_template("evaluation.html", artifact=uidata.load_artifact(),
                               ev=uidata.evaluation_summary())

    @app.route("/health")
    def health():
        return render_template("health.html", artifact=uidata.load_artifact(),
                               health=uidata.data_health())

    return app
