const express = require("express");
const path = require("path");

const app = express();

// Serve all static files (index.html, style.css, main.js) from the current frontend directory
app.use(express.static(__dirname));

app.listen(3000, () => {
    console.log("Frontend Server running at http://localhost:3000");
});