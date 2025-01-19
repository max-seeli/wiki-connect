export function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1)); // Random index from 0 to i
        [array[i], array[j]] = [array[j], array[i]];
    }
    return array;
}

export function splitWordsIntoRows(words) {
    const n = words.length;
    const topSize = Math.floor(n / 3) + (n % 3 === 2 ? 1 : 0);
    const middleSize = Math.floor(n / 3) + (n % 3 >= 1 ? 1 : 0);

    const topRow = words.slice(0, topSize).join(" ");
    const middleRow = words.slice(topSize, topSize + middleSize).join(" ");
    const bottomRow = words.slice(topSize + middleSize).join(" ");

    return [topRow, middleRow, bottomRow].filter(row => row.length > 0);
}

export function addSuggestions(suggestions, datalist) {
    suggestions.forEach(suggestion => {
        const option = document.createElement("option");
        option.value = suggestion;
        datalist.appendChild(option);
    });
};

export function setSvgSize(svg) {
    const width = window.innerWidth,
        height = window.innerHeight;
    svg.attr("width", width).attr("height", height)
        .attr("viewBox", [-width / 2, -height / 2, width, height]);
};

export function updateUrl(id) {
    const url = new URL(window.location);
    url.searchParams.set("name", id);
    window.history.pushState({ id }, "", url);
}

export function getUrlId() {
    const url_params = new URLSearchParams(window.location.search);
    return url_params.get("name");
};

export function addElements(svg, display_nodes, usedEdges, positions, page, clickHandler) {

    svg.append("g")
        .selectAll("line")
        .data(usedEdges)
        .join("line")
        .attr("class", d => `link ${d.class}`)
        .attr("x1", d => positions[d.source].x)
        .attr("y1", d => positions[d.source].y)
        .attr("x2", d => positions[d.target].x)
        .attr("y2", d => positions[d.target].y);

    svg.append("g")
        .selectAll("circle")
        .data(display_nodes)
        .join("circle")
        .attr("class", d => `node ${d.class}`)
        .attr("cx", d => positions[d.id].x)
        .attr("cy", d => positions[d.id].y)
        .on("click", clickHandler)
        .append("title").text(d => d.title);

    let lines = splitWordsIntoRows(page.title.split(" "));
    let dy = 1.2;
    let y = lines.length % 2 == 0 ? -dy * lines.length / 4 : -dy * (lines.length - 1) / 2;
    const label = svg.append("text")
        .attr("x", 0)
        .attr("y", `${y}em`)
        .attr("text-anchor", "middle")
        .attr("fill", "white")
        .attr("font-weight", "bold")
        
    lines.forEach((line, i) => {
        label.append("tspan")
            .attr("x", 0)
            .attr("dy", i === 0 ? 0 : `${dy}em`)
            .text(line);
    });

};
