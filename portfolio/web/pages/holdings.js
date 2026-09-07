/* Holdings page — placeholder. #/holdings/:isin opens the detail drawer
   (pages/holding-detail.js) over this page. Props: { route, snapshot }. */
import { makePlaceholder } from "/static/pages/_placeholder.js";

const HoldingsPage = makePlaceholder({ title: "Holdings", subtitle: "Every position with price provenance, weight and risk share.", icon: "holdings" });
export default HoldingsPage;
