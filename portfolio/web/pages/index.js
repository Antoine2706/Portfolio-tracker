/* Page registry. app.js renders pages[route.page]; the drawer for
   #/holdings/:isin renders `holdingDetail`. Each page is its own module with a
   default export `function XxxPage(props)`; replace files one at a time. */

import OverviewPage from "/static/pages/overview.js";
import HoldingsPage from "/static/pages/holdings.js";
import PerformancePage from "/static/pages/performance.js";
import RiskPage from "/static/pages/risk.js";
import SimulatorPage from "/static/pages/simulator.js";
import InstrumentsPage from "/static/pages/instruments.js";
import TransactionsPage from "/static/pages/transactions.js";
import SettingsPage from "/static/pages/settings.js";
import HoldingDetailPanel from "/static/pages/holding-detail.js";
import KitchenSinkPage from "/static/pages/_kitchensink.js";

export const pages = {
  overview: OverviewPage,
  holdings: HoldingsPage,
  performance: PerformancePage,
  risk: RiskPage,
  simulator: SimulatorPage,
  instruments: InstrumentsPage,
  transactions: TransactionsPage,
  settings: SettingsPage,
  kitchensink: KitchenSinkPage,
  holdingDetail: HoldingDetailPanel,
};

export default pages;
