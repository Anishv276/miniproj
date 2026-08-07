// 1. 3D Tilt Interaction
document.querySelectorAll('.tilt-card').forEach(card => {
    card.addEventListener('mousemove', e => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const centerX = rect.width / 2;
        const centerY = rect.height / 2;

        const rotateX = ((y - centerY) / centerY) * -5;
        const rotateY = ((x - centerX) / centerX) * 5;

        card.style.transform =
            `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.02,1.02,1.02)`;
    });

    card.addEventListener('mouseleave', () => {
        card.style.transform =
            `perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1,1,1)`;
    });
});

// UI Elements
const dropZone = document.getElementById("drop-zone");
const uploadPrompt = document.getElementById("upload-prompt");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const fileSize = document.getElementById("file-size");
const removeFileBtn = document.getElementById("remove-file");

const previewSection = document.getElementById("preview-section");
const imagePreview = document.getElementById("image-preview");
const scanner = document.getElementById("scanner");
const predictBtn = document.getElementById("predict-btn");

const resultsSection = document.getElementById("results-section");
const confidenceBar = document.getElementById("confidence-bar");
const predictionText = document.getElementById("prediction-text");
const confidencePercentage = document.getElementById("confidence-percentage");

let selectedFile = null;

// Disable button initially
predictBtn.disabled = true;

// Drag & Drop
dropZone.addEventListener("click", (e) => {
    if (e.target.closest('#remove-file') || e.target === fileInput) return;
    fileInput.click();
});

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", e => {
    if (e.target.files.length) handleFileSelect(e.target.files[0]);
});

removeFileBtn.addEventListener("click", e => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";

    uploadPrompt.classList.remove("hidden");
    fileInfo.classList.add("hidden");
    previewSection.classList.add("hidden");
    resultsSection.classList.add("hidden");

    confidenceBar.style.width = "0%";
    predictBtn.disabled = true;
});

// File select handler
function handleFileSelect(file) {
    if (!file.type.startsWith("image/")) return;

    selectedFile = file;

    fileName.textContent = file.name;
    fileSize.textContent = (file.size / 1024).toFixed(1) + " KB";

    uploadPrompt.classList.add("hidden");
    fileInfo.classList.remove("hidden");

    const reader = new FileReader();
    reader.onload = e => {
        imagePreview.src = e.target.result;

        previewSection.classList.remove("hidden");
        resultsSection.classList.add("hidden");

        confidenceBar.style.width = "0%";
        predictBtn.disabled = false;
    };

    reader.readAsDataURL(file);
}

// Prediction logic
predictBtn.addEventListener("click", async () => {

    if (!selectedFile) {
        alert("Please upload a fingerprint image first");
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = "Analyzing Patterns...";

    resultsSection.classList.add("hidden");
    confidenceBar.style.width = "0%";

    scanner.classList.remove("hidden");
    scanner.classList.add("scanning-anim");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed");
        }

        await new Promise(resolve => setTimeout(resolve, 2000));

        predictionText.textContent = `Predicted: ${data.predicted}`;
        confidencePercentage.textContent = `${data.confidence.toFixed(2)}%`;

        scanner.classList.add("hidden");
        scanner.classList.remove("scanning-anim");

        predictBtn.disabled = false;
        predictBtn.textContent = "Initialize Scan";

        setTimeout(() => {
            confidenceBar.style.width = `${data.confidence}%`;
        }, 100);

        resultsSection.classList.remove("hidden");
        resultsSection.classList.add("fade-in");

        renderNeonCharts(
            data.labels,
            data.probabilities,
            data.predicted,
            data.confidence
        );

    } catch (error) {
        scanner.classList.add("hidden");
        scanner.classList.remove("scanning-anim");

        predictBtn.disabled = false;
        predictBtn.textContent = "Initialize Scan";

        alert(error.message || "Invalid image or server error");
    }
});

// Chart instances
let barChartInstance = null;
let pieChartInstance = null;

Chart.defaults.color = 'rgba(255,255,255,0.7)';
Chart.defaults.font.family = "'Outfit', sans-serif";

// Charts
function renderNeonCharts(labelsArray, probArray, predictedName, confidenceVal) {
    const barCtx = document.getElementById('barChart').getContext('2d');
    const pieCtx = document.getElementById('pieChart').getContext('2d');

    if (barChartInstance) barChartInstance.destroy();
    if (pieChartInstance) pieChartInstance.destroy();

    const barGradient = barCtx.createLinearGradient(0,0,0,300);
    barGradient.addColorStop(0, '#ff3344');
    barGradient.addColorStop(1, '#4a0404');

    const barHighlight = barCtx.createLinearGradient(0,0,0,300);
    barHighlight.addColorStop(0, '#ff0022');
    barHighlight.addColorStop(1, '#b81414');

    barChartInstance = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: labelsArray,
            datasets: [{
                data: probArray,
                backgroundColor: probArray.map(v => v > 50 ? barHighlight : barGradient),
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100 },
                x: { grid: { display: false } }
            },
            plugins: { legend: { display: false } }
        }
    });

    // 🔥 CENTER TEXT PLUGIN
    const centerTextPlugin = {
        id: 'centerText',
        beforeDraw(chart) {
            const { width, height, ctx } = chart;
            ctx.restore();

            ctx.font = "bold 28px Outfit";
            ctx.fillStyle = "#ffffff";
            ctx.textBaseline = "middle";

            const text = confidenceVal.toFixed(1) + "%";
            const textX = (width - ctx.measureText(text).width) / 2;
            const textY = height / 2;

            ctx.fillText(text, textX, textY);
            ctx.save();
        }
    };

    pieChartInstance = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: [predictedName, 'Others'],
            datasets: [{
                data: [confidenceVal, 100 - confidenceVal],
                backgroundColor: ['#ff0022', 'rgba(255,255,255,0.1)']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: {
                    labels: { color: "#fff" }
                }
            }
        },
        plugins: [centerTextPlugin]
    });
}