/* Allocate — "where should this purchase go". Props: { route, snapshot }.

   The one page that answers a decision rather than describing a state. Type an
   amount; get whole shares per holding, what it costs, and how much of the gap
   to equal risk contribution it actually closes.

   Reading order, and it is the argument rather than a layout:
   1. The headline is the smallest purchase that meaningfully changes anything.
      If you type 200 EUR and the honest answer is that the destination does not
      matter, that sentence is the first thing on the page — before any table
      that would imply it does.
   2. Four floors, not one. Where the book is, where this purchase lands it, the
      best reachable WITH THIS MUCH MONEY, and the best if selling were allowed.
      An improvement shown on its own is a result; shown against a floor it
      cannot pass, it is a fact about the account.
   3. The order: whole shares, per broker, with risk share before → after.
   4. What each destination would cost against what it would buy, so a
      wide-spread holding cannot quietly consume the improvement it delivers.
   5. What was refused, and why. Never folded away.

   The page computes nothing. Every figure comes from POST /api/allocate, which
   is the same `agents.allocate` code the CLI runs and the replay evaluates; a
   second implementation for the screen would be a second answer to defend.
   Submitted on demand rather than debounced, because unlike the simulator this
   is a solve rather than a projection and it takes long enough to notice.

   It never proposes a sale, never proposes an amount you did not type, and
   makes no claim about return. It has no view on returns and neither do we. */

import { html, useState, useMemo, useRef } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { Button } from "/static/components/Button.js";
import { Banner } from "/static/components/Banner.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { DataTable } from "/static/components/DataTable.js";
import { KpiTile } from "/static/components/KpiTile.js";
import { Field, Input } from "/static/components/Field.js";
import { Help } from "/static/components/Tooltip.js";
import { allocate, errorMessage } from "/static/lib/api.js";
import { parseAmount } from "/static/lib/amount.js";
import * as fmt from "/static/lib/format.js";

const SUBTITLE = "New money is the one rebalancing channel that costs nothing extra, because the purchase was going to happen anyway. This chooses where it goes. It never proposes a sale.";

/** The last amount and answer, kept for the session so returning restores them. */
let remembered = null;

/* ---------------- the headline ---------------- */

/** The first sentence, which is sometimes "this hardly moves anything".

    Deliberately above the order rather than beneath it. A table of destinations
    printed first implies the choice is worth making; saying otherwise afterwards
    is saying it too late.

    Two questions, and they are kept apart because merging them once printed
    "where this goes matters" directly above "closes 2.3% of the gap". How much
    the purchase MOVES the book is the threshold; how much the CHOICE is worth is
    the best-worst spread. A purchase too small to matter can still have a
    destination worth avoiding, and both facts belong in the reader's head. */
function Headline({ result, ccy }) {
  const { cash, meaningful_cash: threshold, best_worst_gap: gap } = result;
  const choice = html`<div>Best and worst destinations differ by
    ${" "}${fmt.num(gap, { decimals: 3 })} of dispersion, so the choice itself is worth
    ${gap > 0.01 ? " making" : " almost nothing"}.</div>`;
  if (threshold == null) {
    return html`<${Banner} kind="info" title="Nothing here to even out.">
      <div>No purchase of any size meaningfully changes the risk shares of this book, so put this
      where you want it. The tool has nothing to add.</div>
    <//>`;
  }
  if (cash < threshold) {
    return html`<${Banner} kind="info" title=${`At ${fmt.money(cash, ccy, { decimals: 0 })} this hardly moves the book.`}>
      <div>The best any purchase this size reaches is
      ${" "}${fmt.num(result.floor_at_cash, { decimals: 4 })} against
      ${" "}${fmt.num(result.dispersion_now, { decimals: 4 })} now. The smallest purchase that closes
      a worthwhile share of the gap is about
      ${" "}<span class="strong">${fmt.money(threshold, ccy, { decimals: 0 })}</span>.</div>
      ${choice}
    <//>`;
  }
  return html`<${Banner} kind="good" title="This moves the book.">
    <div>${fmt.money(cash, ccy, { decimals: 0 })} closes
    ${" "}<span class="strong">${fmt.pct(result.closable)}</span> of the gap to equal risk contribution.</div>
    ${choice}
  <//>`;
}

/* ---------------- the four floors ---------------- */

const FLOOR_HELP = {
  now: "Coefficient of variation of the risk shares: their standard deviation over their mean. Zero when every holding carries the same share of the book's volatility, at most √(m−1) for m holdings. This is what the solver minimises and what it reports — the same number, so it cannot grade an answer by a rule it did not use to produce it.",
  spread: "Largest gap between any two risk shares, over the same mean. Descriptive only. It says how far apart the extremes are, which a coefficient of variation does not, but it is decided by two holdings and discards the rest, so nothing optimises it.",
  after: "Recomputed on the whole-share order below, not on the continuous solution. What you would actually own.",
  floor: "The best any buy-only purchase of this size reaches. Buy-only reduces an overweight risk share only by dilution, so this is usually well above zero.",
  unlimited: "What selling could reach: the equal risk contribution portfolio itself. Shown so the buy-only floor is read against something rather than against an assumed zero.",
};

function Floors({ result }) {
  const n = (v) => (v == null ? fmt.DASH : fmt.num(v, { decimals: 4 }));
  return html`<div class="kpi-grid">
    <${KpiTile} label=${html`Dispersion now <${Help} text=${FLOOR_HELP.now} />`} value=${n(result.dispersion_now)}
      sub=${html`range ${n(result.spread_now)} <${Help} text=${FLOOR_HELP.spread} />`} />
    <${KpiTile} label=${html`After this purchase <${Help} text=${FLOOR_HELP.after} />`} value=${n(result.dispersion_after)}
      sub=${`closes ${fmt.pct(result.closable)} of the gap · range ${n(result.spread_after)}`} />
    <${KpiTile} label=${html`Floor at this amount <${Help} text=${FLOOR_HELP.floor} />`} value=${n(result.floor_at_cash)}
      sub=${`whole shares cost ${fmt.num(result.rounding_penalty, { decimals: 4 })}`} />
    <${KpiTile} label=${html`Floor if selling were allowed <${Help} text=${FLOOR_HELP.unlimited} />`} value=${n(result.floor_unlimited)}
      sub="not on offer here" />
  </div>`;
}

/* ---------------- the order ---------------- */

/** "before → after" in one cell, with the change beneath. */
function baCell(before, after, deltaText, deltaClass = "") {
  return html`<span>${before}<span class="faint"> → </span><span class="strong">${after}</span></span>
    ${deltaText ? html`<span class=${["cell-sub", deltaClass].filter(Boolean).join(" ")}>${deltaText}</span>` : null}`;
}

function Order({ result, ccy }) {
  const columns = useMemo(() => [
    { key: "shares", label: "Shares", numeric: true, format: "int",
      render: (r, v) => html`${fmt.int(v)}<span class="cell-sub">@ ${fmt.money(r.price, ccy)}</span>` },
    { key: "name", label: "Instrument", primary: true,
      render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${v}<span class="cell-sub">${r.isin} · ${r.broker}</span></div>` },
    { key: "amount", label: "Amount", numeric: true, format: (v) => fmt.money(v, ccy, { decimals: 0 }),
      render: (r, v) => html`${fmt.money(v, ccy, { decimals: 0 })}<span class="cell-sub">costs ${fmt.money(r.cost, ccy)}</span>` },
    { key: "weight_after", label: "Weight", numeric: true, title: "Share of the book's value, before → after",
      render: (r) => baCell(fmt.pct(r.weight_before), fmt.pct(r.weight_after)) },
    { key: "risk_after", label: "Risk share", numeric: true,
      title: "Share of the book's volatility, before → after. Evening these out is the whole objective.",
      render: (r) => {
        const d = r.risk_after - r.risk_before;
        return baCell(fmt.pct(r.risk_before), fmt.pct(r.risk_after),
                      Math.abs(d) < 1e-9 ? null : fmt.pp(d), "muted");
      } },
  ], [ccy]);

  const parts = Object.entries(result.cost_parts || {}).filter(([, v]) => v);
  const negative = result.purchases.some((p) => Math.min(p.risk_before, p.risk_after) < 0);
  const footer = html`<div class="stack gap-1">
    <div class="row wrap gap-4">
      <span class="stat-inline"><span class="stat-label">Invested</span><span class="stat-value">${fmt.money(result.invested, ccy)}</span></span>
      <span class="stat-inline"><span class="stat-label">Left over</span><span class="stat-value">${fmt.money(result.leftover, ccy)}</span></span>
      <span class="stat-inline"><span class="stat-label">Cost</span><span class="stat-value">${fmt.money(result.total_cost, ccy)}</span></span>
      ${parts.map(([k, v]) => html`<span key=${k} class="stat-inline"><span class="stat-label">${k}</span><span class="stat-value">${fmt.money(v, ccy)}</span></span>`)}
    </div>
    ${negative ? html`<div class="small muted">A risk share below zero is not a misprint. A holding that moves against the
      rest of the book has a negative marginal contribution to its volatility, so it carries risk away rather than adding
      it — that is what a hedge is for, and the shares still sum to 100%.</div>` : null}
    <div class="small muted">Whole shares only, at the latest delayed price. Nothing here is written to the ledger and no
      order is placed anywhere: this is a list to type into your broker yourself.</div>
  </div>`;

  return html`<${Card} title="The order" flush footer=${footer}
    caption=${`${fmt.count(result.purchases.length, "line")} · left over is what whole shares could not spend`}>
    <${DataTable} columns=${columns} rows=${result.purchases} rowKey="isin"
      sort=${{ key: "amount", dir: "desc" }} empty="Nothing to buy at this amount" />
  <//>`;
}

/* ---------------- destinations, refusals, the inverse question ---------------- */

function Destinations({ result, ccy }) {
  const columns = useMemo(() => [
    { key: "name", label: "If it all went here", primary: true,
      render: (r, v) => html`<div class="truncate" style="max-width:240px" title=${v}>${v}<span class="cell-sub">${r.isin}</span></div>` },
    { key: "dispersion", label: "Dispersion", numeric: true, format: (v) => fmt.num(v, { decimals: 4 }) },
    { key: "improvement", label: "Improvement", numeric: true, format: (v) => fmt.num(v, { decimals: 4 }) },
    { key: "cost", label: "Cost", numeric: true,
      render: (r, v) => (v == null
        ? html`<span class="faint" title="This broker has no published fee for an order this size, so it is refused rather than priced">${fmt.DASH}</span>`
        : fmt.money(v, ccy)) },
    { key: "per_euro", label: html`Gain per EUR <${Help} text="Dispersion closed per euro of trading cost. A wide-spread holding can deliver a real improvement and charge more than it is worth." />`,
      numeric: true,
      render: (r, v) => (v == null ? html`<span class="faint">${fmt.DASH}</span>` : fmt.num(v, { decimals: 5 })) },
  ], [ccy]);
  return html`<${Card} title="What each destination is worth" flush
    caption="The whole amount into one holding, costed. Not a recommendation — the order above is the answer; this is what it is being chosen against.">
    <${DataTable} columns=${columns} rows=${result.destinations} rowKey="isin"
      sort=${{ key: "dispersion", dir: "asc" }} empty="Nothing is buyable" />
  <//>`;
}

function Refused({ result }) {
  if (!result.refused.length) return null;
  return html`<${Card} title="Refused" caption="Named rather than silently dropped: a holding missing from the order for an unstated reason is a holding you would go and buy anyway.">
    <ul class="stack gap-1">
      ${result.refused.map((r) => html`<li key=${r.isin}><span class="strong">${r.name}</span>${" "}<span class="faint">${r.isin}</span><div class="small muted">${r.reason}</div></li>`)}
    </ul>
  <//>`;
}

/** The inverse question, which is usually the decision-relevant one. */
function Target({ result, ccy }) {
  if (result.target == null) return null;
  const pinned = result.pinned || [];
  const caveat = !pinned.length ? null : html`<div class="small muted">
    ${fmt.count(pinned.length, "holding")} cannot receive new money (${pinned.join(", ")}), so past some amount more cash
    dilutes ${pinned.length > 1 ? "them" : "it"} towards a zero risk share faster than it evens the rest out, and the floor
    starts rising again. Every amount named here does reach what it claims; what that costs is the guarantee that nothing
    smaller would.</div>`;
  const body = result.cash_for_target != null
    ? html`<div>Reaching a dispersion of ${fmt.num(result.target, { decimals: 2 })} buy-only would take about
        ${" "}<span class="strong">${fmt.money(result.cash_for_target, ccy, { decimals: 0 })}</span> of new money,
        against a book of ${fmt.money(result.book_value, ccy, { decimals: 0 })}.</div>`
    : html`<div>No purchase reaches a dispersion of ${fmt.num(result.target, { decimals: 2 })} buy-only, at any size this
        tool searched. The best it found was <span class="strong">${fmt.num(result.best_reachable, { decimals: 2 })}</span>,
        at about ${fmt.money(result.best_reachable_cash, ccy, { decimals: 0 })}. Reaching your target means selling, not
        contributing.</div>`;
  return html`<${Card} title="What would that take?" class="stack gap-1">${body}${caveat}<//>`;
}

/** What the typed string was understood to mean, shown under the field.

    Two jobs. On an unambiguous entry it echoes the amount back formatted, so
    a misread is visible before the button is pressed rather than inferred
    afterwards from a strange answer. On an ambiguous one -- "10.000", which
    is ten thousand in Brussels and ten in London -- it refuses and offers
    both readings as buttons.

    The refusal is the part that matters. The previous parser read "10.000"
    as ten, silently, and the page then proposed buying one share: no error,
    no warning, and a result that looks like a broken page rather than a
    misread number. */
function Reading({ read, ccy, onPick }) {
  if (read.state === "empty") return null;
  if (read.state === "invalid") {
    return html`<div class="field-hint" role="status">That is not a number.</div>`;
  }
  if (read.state === "ambiguous") {
    const [big, small] = read.readings;
    return html`<div class="field-hint" role="status">
      Ambiguous:${" "}<strong>${fmt.money(big, ccy, { decimals: 0 })}</strong> or${" "}
      <strong>${fmt.money(small, ccy)}</strong>? A dot or comma with three
      digits after it is a thousands separator in some places and a decimal
      point in others, and nothing here settles which you meant.${" "}
      <${Button} size="sm" variant="secondary" onClick=${() => onPick(big)}>
        ${fmt.money(big, ccy, { decimals: 0 })}<//>${" "}
      <${Button} size="sm" variant="secondary" onClick=${() => onPick(small)}>
        ${fmt.money(small, ccy)}<//>
    </div>`;
  }
  // htm collapses the newline before an interpolation, which is how this
  // project previously shipped "14.7135against" and "WheatJE00BN7KB664".
  return html`<div class="field-hint" role="status">Understood as${" "}
    <strong>${fmt.money(read.value, ccy)}</strong>.</div>`;
}


/* ---------------- the page ---------------- */

export default function AllocatePage({ snapshot }) {
  const ccy = (snapshot && snapshot.totals && snapshot.totals.currency) || "EUR";
  const [amount, setAmount] = useState(remembered ? remembered.amount : "");
  const [target, setTarget] = useState(remembered ? remembered.target : "");
  const [result, setResult] = useState(remembered ? remembered.result : null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState(null);
  const seq = useRef(0);

  const read = parseAmount(amount);
  const readTarget = parseAmount(target);
  const cash = read.value;
  const wanted = readTarget.value;
  const valid = cash != null && cash > 0;

  const run = async () => {
    if (!valid) { setError({ message: "An amount to invest is needed, and it has to be positive." }); return; }
    const mine = ++seq.current;
    setPending(true);
    setError(null);
    try {
      const body = wanted != null ? { amount: cash, target: wanted } : { amount: cash };
      const answer = await allocate(body);
      if (mine !== seq.current) return;           // a later request already won
      setResult(answer);
      remembered = { amount, target, result: answer };
    } catch (err) {
      if (mine !== seq.current) return;
      setError({ status: err && err.status, message: errorMessage(err) });
    } finally {
      if (mine === seq.current) setPending(false);
    }
  };

  const onKey = (e) => { if (e.key === "Enter") run(); };

  const risk = snapshot && snapshot.risk;
  if (!snapshot) {
    return html`<${Page} title="Allocate" subtitle=${SUBTITLE}>
      <${Card}><${EmptyState} icon="wallet" title="Loading the book…" body="Directing new money needs the holdings and the risk model." /><//>
    <//>`;
  }
  if (!risk || !risk.available) {
    return html`<${Page} title="Allocate" subtitle=${SUBTITLE}>
      <${Card}><${EmptyState} icon="risk" title="No risk model to allocate against"
        body=${(risk && risk.reason) || "Without a covariance matrix there is no risk structure to even out, so any destination is as good as any other."}
        action=${html`<${Button} variant="secondary" href="#/risk">Open Risk<//>`} /><//>
    <//>`;
  }

  const errorBanner = !error ? null : error.status === 409
    ? html`<${Banner} kind="warn" title="No risk model right now."
        action=${html`<${Button} size="sm" variant="secondary" href="#/risk">Open Risk<//>`}>${error.message}<//>`
    : html`<${Banner} kind="error" title=${error.status === 400 ? "The server refused the request." : "The allocation failed."}
        action=${html`<${Button} size="sm" variant="secondary" onClick=${run}>Try again<//>`}>${error.message}<//>`;

  return html`<${Page} title="Allocate" subtitle=${SUBTITLE}>
    <div class="grid">
      <${Card} class="col-12" title="How much are you putting in?"
        caption="Whole shares only. No sales are ever proposed, and no amount you have not typed.">
        <div class="row wrap gap-4">
          <${Field} label="New money" id="alloc-amount" hint="The lump you are about to invest.">
            <${Input} id="alloc-amount" numeric autoFocus value=${amount} placeholder="5000" prefix="€"
              onInput=${(e) => setAmount(e.target.value)} onKeyDown=${onKey} />
            <${Reading} read=${read} ccy=${ccy} onPick=${(v) => setAmount(String(v))} />
          <//>
          <${Field} id="alloc-target" hint="Optional. Answers: what would reaching that take?"
            label=${html`Target dispersion <${Help} text="The inverse question, and usually the decision-relevant one. If reaching 1.0 would take 40,000 EUR on a 17,000 EUR book, the structure cannot be fixed by contributions and the real choice is whether to sell." />`}>
            <${Input} id="alloc-target" numeric value=${target} placeholder="1.0"
              onInput=${(e) => setTarget(e.target.value)} onKeyDown=${onKey} />
          <//>
          <${Button} variant="primary" icon="zap" onClick=${run} disabled=${!valid || pending} loading=${pending}>
            ${pending ? "Solving…" : "Where should it go?"}
          <//>
        </div>
        ${errorBanner}
      <//>

      ${!result ? html`<${Card} class="col-12"><${EmptyState} icon="wallet"
          title=${pending ? "Solving…" : "Type an amount"}
          body="The answer is whole shares per holding, what the purchase costs, and how much of the gap to equal risk contribution it actually closes — including when the answer is that it hardly matters where the money goes." /><//>`
      : html`
        <div class="col-12">
          <div class="stack gap-4">
            <${Headline} result=${result} ccy=${ccy} />
            <${Floors} result=${result} />
            <${Target} result=${result} ccy=${ccy} />
            <${Order} result=${result} ccy=${ccy} />
            <${Destinations} result=${result} ccy=${ccy} />
            <${Refused} result=${result} />
            ${result.assumptions.length ? html`<${Card} title="Cost inputs that are estimates"
              caption="Every input is either read off a contract note or marked as an estimate. These are the estimates.">
              <ul class="stack gap-1">${result.assumptions.map((a, i) => html`<li key=${i} class="small muted">${a}</li>`)}</ul>
            <//>` : null}
          </div>
        </div>`}
    </div>
  <//>`;
}
