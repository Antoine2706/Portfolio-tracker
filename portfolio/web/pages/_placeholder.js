/* Placeholder page factory used until each page is built.
   makePlaceholder({ title, subtitle, icon }) → a page component. */

import { html } from "/static/vendor/preact-htm.module.js";
import { Page } from "/static/components/Page.js";
import { Card } from "/static/components/Card.js";
import { EmptyState } from "/static/components/EmptyState.js";
import { Button } from "/static/components/Button.js";
import { navigate } from "/static/lib/router.js";

export function makePlaceholder({ title, subtitle, icon = "hammer" }) {
  return function PlaceholderPage() {
    return html`<${Page} title=${title} subtitle=${subtitle}>
      <${Card}>
        <${EmptyState} icon=${icon} title="Being built"
          body=${`The ${title} page is on its way. The shell, data and chart infrastructure are in place — see the component gallery for what it will be built from.`}
          action=${html`<${Button} variant="secondary" icon="layers" onClick=${() => navigate("/kitchensink")}>Open component gallery<//>`} />
      <//>
    <//>`;
  };
}
