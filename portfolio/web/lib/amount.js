/* Reading a money amount someone typed, in a place where "10.000" is
   ambiguous and guessing wrong costs a factor of a thousand.

   Why this file exists
   --------------------
   `allocate.js` and `simulator.js` each had their own copy of a parser that
   read "10.000" as ten. In Belgium that is how ten thousand is written, so
   typing the amount of a real purchase produced an allocation of one share
   and a page that looked broken. There was no error: the number parsed
   cleanly, it was simply the wrong number.

   `data/importers.py::parse_number` already had this right for CSV import.
   The web layer had diverged from it, twice, in the direction that fails
   silently.

   The rule
   --------
   Both separators present: the LAST one is the decimal point, because no
   convention puts the thousands separator after the decimal one. So
   "1.250,50" is 1250.50 and "1,250.50" is the same number.

   One kind of separator, appearing more than once: thousands separators.
   "1.234.567" is unambiguous -- no convention has two decimal points.

   One separator, with a tail that is NOT three digits: a decimal separator.
   "5000,5" and "10.5" are unambiguous, because a thousands group is always
   exactly three digits.

   One separator, with a tail of exactly three digits: **ambiguous, and
   refused**. "10.000" is ten thousand to half of Europe and ten to the
   other half, and nothing in the string settles it. This is the same
   judgement the CLI already makes about a bare "0.12" for a tax rate: a
   hundredfold error from one keystroke has nothing to notice it by, so the
   answer is to ask rather than to pick.

   Refusing is not the whole fix, because a refusal the user does not
   understand is just a different way of not working. So the parse also
   returns both readings, and the caller shows them: "did you mean EUR 10,000
   or EUR 10.00?" is answerable in one click, where a silent EUR 10 is not
   noticeable at all.
*/

/** The result of reading a typed amount.
 *
 *  `value`     the number, or null if there isn't one
 *  `state`     "empty" | "ok" | "ambiguous" | "invalid"
 *  `readings`  for "ambiguous": the two numbers it could be, larger first
 */
export function parseAmount(text) {
  if (text == null) return { value: null, state: "empty", readings: [] };

  // Strip spaces (including the narrow no-break space a browser may insert
  // from a locale-formatted paste) and any currency decoration.
  const cleaned = String(text)
    .replace(/[\s  ]/g, "")
    .replace(/€|EUR|£|\$/gi, "")
    .trim();
  if (!cleaned) return { value: null, state: "empty", readings: [] };
  if (!/^[+-]?[0-9.,]+$/.test(cleaned)) {
    return { value: null, state: "invalid", readings: [] };
  }

  const sign = cleaned.startsWith("-") ? -1 : 1;
  const body = cleaned.replace(/^[+-]/, "");
  if (!body || !/[0-9]/.test(body)) {
    return { value: null, state: "invalid", readings: [] };
  }

  const commas = (body.match(/,/g) || []).length;
  const dots = (body.match(/\./g) || []).length;

  const asDecimal = (sep) => {
    const other = sep === "," ? "." : ",";
    const t = body.split(other).join("").replace(sep, ".");
    return Number(t);
  };
  const asThousands = () => Number(body.replace(/[.,]/g, ""));

  let value = null;
  if (commas && dots) {
    // The last separator to appear is the decimal point.
    value = body.lastIndexOf(",") > body.lastIndexOf(".")
      ? asDecimal(",") : asDecimal(".");
  } else if (commas + dots === 0) {
    value = Number(body);
  } else if (commas > 1 || dots > 1) {
    value = asThousands();                       // "1.234.567"
  } else {
    const sep = commas ? "," : ".";
    const tail = body.slice(body.lastIndexOf(sep) + 1);
    if (tail.length === 3) {
      // "10.000": ten thousand, or ten? Nothing here can tell.
      const big = asThousands();
      const small = asDecimal(sep);
      if (!Number.isFinite(big) || !Number.isFinite(small)) {
        return { value: null, state: "invalid", readings: [] };
      }
      return { value: null, state: "ambiguous", readings: [sign * big, sign * small] };
    }
    value = asDecimal(sep);
  }

  if (!Number.isFinite(value)) return { value: null, state: "invalid", readings: [] };
  return { value: sign * value, state: "ok", readings: [] };
}

/** Just the number, or null. For callers that only need the value. */
export function amountValue(text) {
  return parseAmount(text).value;
}
