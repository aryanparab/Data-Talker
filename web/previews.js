(async () => {
  const host = document.getElementById("dataset-previews");
  if (!host) return;
  const make = (tag, text, cls) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (cls) node.className = cls;
    return node;
  };
  try {
    const response = await fetch("/api/catalog");
    if (!response.ok) throw new Error("Unavailable");
    const data = await response.json();
    const fragment = document.createDocumentFragment();
    Object.entries(data.tables).forEach(([name, table]) => {
      const card = make("article", undefined, "sample-table-card");
      const heading = make("div", undefined, "sample-table-heading");
      heading.append(
        make("h3", name),
        make("span", `${table.rows.toLocaleString()} rows`),
      );
      card.append(heading);
      const scroll = make("div", undefined, "sample-table-scroll");
      scroll.tabIndex = 0;
      scroll.setAttribute("role", "region");
      scroll.setAttribute("aria-label", name + " sample rows");
      const grid = make("table");
      grid.append(
        make("caption", `First five rows of ${name}, ordered by all columns.`),
      );
      const header = make("thead"),
        tr = make("tr");
      Object.keys(table.columns).forEach((column) => {
        const th = make("th", column);
        th.scope = "col";
        tr.append(th);
      });
      header.append(tr);
      grid.append(header);
      const body = make("tbody");
      (data.previews[name] || []).forEach((row) => {
        const tr = make("tr");
        row.forEach((value, index) =>
          tr.append(
            make(
              "td",
              value === null
                ? "NULL"
                : typeof value === "number" &&
                    /DECIMAL|FLOAT|DOUBLE/.test(
                      Object.values(table.columns)[index],
                    )
                  ? new Intl.NumberFormat("en-US", {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    }).format(value)
                  : String(value),
            ),
          ),
        );
        body.append(tr);
      });
      grid.append(body);
      scroll.append(grid);
      card.append(scroll);
      fragment.append(card);
    });
    host.replaceChildren(fragment);
  } catch {
    host.replaceChildren(
      make(
        "p",
        "Table previews are temporarily unavailable. Please refresh in a moment.",
        "preview-loading",
      ),
    );
  }
})();
