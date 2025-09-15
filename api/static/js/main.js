// ===== Global Variables =====
let availableModels = [];
let currentTheme = localStorage.getItem("theme") || "light";

// ===== Theme Management =====
function initializeTheme() {
  document.documentElement.setAttribute("data-theme", currentTheme);
  updateThemeIcon();
}

function toggleTheme() {
  currentTheme = currentTheme === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", currentTheme);
  localStorage.setItem("theme", currentTheme);
  updateThemeIcon();
}

function updateThemeIcon() {
  const icon = document.querySelector(".theme-toggle i");
  icon.className = currentTheme === "light" ? "fas fa-moon" : "fas fa-sun";
}

// ===== Tab Management =====
function switchTab(tabName) {
  // Remove active class from all tabs and content
  document
    .querySelectorAll(".tab")
    .forEach((tab) => tab.classList.remove("active"));
  document
    .querySelectorAll(".tab-content")
    .forEach((content) => content.classList.remove("active"));

  // Add active class to selected tab and content
  event.target.classList.add("active");
  document.getElementById(tabName + "-tab").classList.add("active");
}

// ===== Model Loading =====
async function loadModels() {
  try {
    const response = await fetch("/api/models");
    if (response.ok) {
      const data = await response.json();
      availableModels = data.models || [];
      populateModelSelect();
      displayModelGrid();
    } else {
      console.error("Failed to load models");
      displayModelError();
    }
  } catch (error) {
    console.error("Error loading models:", error);
    displayModelError();
  }
}

function populateModelSelect() {
  const select = document.getElementById("basic-model");
  select.innerHTML = "";

  availableModels.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    select.appendChild(option);
  });

  if (availableModels.length > 0) {
    select.selectedIndex = 0;
  }
}

function displayModelGrid() {
  const grid = document.getElementById("model-grid");
  grid.className = "model-grid"; // Add the CSS class
  grid.innerHTML = "";

  if (availableModels.length === 0) {
    grid.innerHTML =
      '<div class="alert alert-warning">No models available</div>';
    return;
  }

  availableModels.forEach((model) => {
    const card = document.createElement("div");
    card.className = "model-card";
    card.innerHTML = `
            <h3><i class="fas fa-robot"></i> ${model.name}</h3>
            <p>${
              model.description ||
              "Advanced machine learning model for fake news detection"
            }</p>
            <div class="feature-value" style="margin-top: 0.5rem; color: var(--success-color);">
                <i class="fas fa-check-circle"></i> Ready
            </div>
        `;
    grid.appendChild(card);
  });
}

function displayModelError() {
  const grid = document.getElementById("model-grid");
  grid.innerHTML = `
        <div class="alert alert-error">
            <i class="fas fa-exclamation-triangle"></i>
            Failed to load models. Please check server connection.
        </div>
    `;
}

// ===== Sample Data =====
const sampleData = {
  fake: {
    title: "BREAKING: Scientists Discover Shocking Truth About Coffee",
    text: "In a groundbreaking study that will change everything you thought you knew about coffee, researchers have discovered that drinking coffee backwards can increase your IQ by 500%. The study, conducted by the prestigious Institute of Alternative Facts, tested over 10,000 participants who drank their morning coffee while standing on their heads. 'The results were absolutely astounding,' said Dr. Jane Doe, lead researcher. 'Participants showed immediate improvements in everything from memory to telepathic abilities.' The coffee industry is reportedly in panic mode as this revolutionary discovery threatens to completely revolutionize how we consume our daily caffeine.",
    source: "alternativefacts.com",
  },
  real: {
    title: "New Study Links Regular Exercise to Improved Mental Health",
    text: "A comprehensive study published in the Journal of Health Psychology has found a strong correlation between regular physical exercise and improved mental health outcomes. The research, which followed 5,000 participants over two years, showed that individuals who engaged in at least 150 minutes of moderate exercise per week reported significantly lower levels of anxiety and depression. Dr. Sarah Johnson, the study's lead author, noted that the benefits were consistent across different age groups and fitness levels. 'The data clearly shows that even modest increases in physical activity can have meaningful impacts on psychological well-being,' Johnson stated. The study reinforces existing guidelines from health organizations worldwide that recommend regular exercise as part of a healthy lifestyle.",
    source: "reuters.com",
  },
};

function loadSample(type) {
  const sample = sampleData[type];
  if (!sample) return;

  // Load into basic tab
  document.getElementById("basic-text").value = sample.text;

  // Load into enhanced tab
  document.getElementById("enhanced-title").value = sample.title;
  document.getElementById("enhanced-text").value = sample.text;
  document.getElementById("enhanced-source").value = sample.source;

  // Load into features tab
  document.getElementById("feature-title").value = sample.title;
  document.getElementById("feature-text").value = sample.text;
  document.getElementById("feature-source").value = sample.source;

  showNotification(
    `Loaded ${type} news sample`,
    type === "fake" ? "warning" : "success"
  );
}

// ===== Prediction Functions =====
async function predictBasic() {
  const text = document.getElementById("basic-text").value.trim();
  const modelSelect = document.getElementById("basic-model");
  const modelId = modelSelect.value;

  if (!text) {
    showNotification("Please enter some text to analyze", "error");
    return;
  }

  if (!modelId) {
    showNotification("Please select a model", "error");
    return;
  }

  const button = event.target;
  const originalText = button.innerHTML;
  setButtonLoading(button, "Analyzing...");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        model: modelId,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      displayBasicResult(data);
      showNotification("Analysis complete!", "success");
    } else {
      showNotification(data.error || "Analysis failed", "error");
    }
  } catch (error) {
    console.error("Prediction error:", error);
    showNotification("Network error occurred", "error");
  } finally {
    setButtonLoading(button, originalText, false);
  }
}

async function predictEnhanced() {
  const title = document.getElementById("enhanced-title").value.trim();
  const text = document.getElementById("enhanced-text").value.trim();
  const source = document.getElementById("enhanced-source").value.trim();
  const url = document.getElementById("enhanced-url").value.trim();

  if (!text) {
    showNotification("Please enter article text", "error");
    return;
  }

  const button = event.target;
  const originalText = button.innerHTML;
  setButtonLoading(button, "Analyzing...");

  try {
    const response = await fetch("/predict/enhanced", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title,
        text: text,
        source: source,
        url: url,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      displayEnhancedResult(data);
      showNotification("Enhanced analysis complete!", "success");
    } else {
      showNotification(data.error || "Analysis failed", "error");
    }
  } catch (error) {
    console.error("Enhanced prediction error:", error);
    showNotification("Network error occurred", "error");
  } finally {
    setButtonLoading(button, originalText, false);
  }
}

async function analyzeFeatures() {
  const title = document.getElementById("feature-title").value.trim();
  const text = document.getElementById("feature-text").value.trim();
  const source = document.getElementById("feature-source").value.trim();

  if (!text) {
    showNotification("Please enter text to analyze", "error");
    return;
  }

  const button = event.target;
  const originalText = button.innerHTML;
  setButtonLoading(button, "Extracting...");

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: title,
        text: text,
        source: source,
      }),
    });

    const data = await response.json();

    if (response.ok) {
      displayFeaturesResult(data);
      showNotification("Feature analysis complete!", "success");
    } else {
      showNotification(data.error || "Analysis failed", "error");
    }
  } catch (error) {
    console.error("Feature analysis error:", error);
    showNotification("Network error occurred", "error");
  } finally {
    setButtonLoading(button, originalText, false);
  }
}

// ===== Result Display Functions =====
function displayBasicResult(data) {
  const resultDiv = document.getElementById("basic-result");
  const prediction = data.prediction;
  const confidence = data.confidence;
  const model = data.model_used;

  const isReal = prediction.toLowerCase() === "real";
  const confidencePercent = Math.round(confidence * 100);

  resultDiv.innerHTML = `
        <div class="result-card">
            <h3>
                <i class="fas ${
                  isReal ? "fa-check-circle" : "fa-exclamation-triangle"
                }"></i>
                Prediction: <span style="color: ${
                  isReal ? "var(--success-color)" : "var(--danger-color)"
                }">${prediction}</span>
            </h3>
            <div class="confidence-bar">
                <div class="confidence-fill ${isReal ? "real" : "fake"}" 
                     style="width: ${confidencePercent}%"></div>
            </div>
            <p>Confidence: <strong>${confidencePercent}%</strong></p>
            <p>Model: <strong>${model}</strong></p>
            <div class="alert ${
              isReal ? "alert-success" : "alert-warning"
            }" style="margin-top: 1rem;">
                <i class="fas ${
                  isReal ? "fa-info-circle" : "fa-exclamation-triangle"
                }"></i>
                ${
                  isReal
                    ? "This article appears to be legitimate news content."
                    : "This article shows characteristics commonly found in fake news."
                }
            </div>
        </div>
    `;
}

function displayEnhancedResult(data) {
  const resultDiv = document.getElementById("enhanced-result");
  const prediction = data.prediction;
  const confidence = data.confidence;
  const features = data.features || {};

  const isReal = prediction.toLowerCase() === "real";
  const confidencePercent = Math.round(confidence * 100);

  let featuresHtml = "";
  if (Object.keys(features).length > 0) {
    featuresHtml = `
            <div class="features-grid">
                ${Object.entries(features)
                  .map(
                    ([key, value]) => `
                    <div class="feature-item">
                        <h4>${formatFeatureName(key)}</h4>
                        <div class="feature-value">${formatFeatureValue(
                          key,
                          value
                        )}</div>
                    </div>
                `
                  )
                  .join("")}
            </div>
        `;
  }

  resultDiv.innerHTML = `
        <div class="result-card">
            <h3>
                <i class="fas ${
                  isReal ? "fa-shield-check" : "fa-shield-exclamation"
                }"></i>
                Enhanced Prediction: <span style="color: ${
                  isReal ? "var(--success-color)" : "var(--danger-color)"
                }">${prediction}</span>
            </h3>
            <div class="confidence-bar">
                <div class="confidence-fill ${isReal ? "real" : "fake"}" 
                     style="width: ${confidencePercent}%"></div>
            </div>
            <p>Overall Confidence: <strong>${confidencePercent}%</strong></p>
            ${featuresHtml}
            <div class="alert ${
              isReal ? "alert-success" : "alert-warning"
            }" style="margin-top: 1rem;">
                <i class="fas ${isReal ? "fa-thumbs-up" : "fa-flag"}"></i>
                ${
                  isReal
                    ? "Enhanced analysis confirms this appears to be legitimate news."
                    : "Enhanced analysis suggests this content may be misleading or false."
                }
            </div>
        </div>
    `;
}

function displayFeaturesResult(data) {
  const resultDiv = document.getElementById("features-result");
  const features = data.features || {};

  if (Object.keys(features).length === 0) {
    resultDiv.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle"></i>
                No features extracted. Please try with different content.
            </div>
        `;
    return;
  }

  const featuresHtml = Object.entries(features)
    .map(
      ([key, value]) => `
        <div class="feature-item">
            <h4><i class="fas ${getFeatureIcon(key)}"></i> ${formatFeatureName(
        key
      )}</h4>
            <div class="feature-value">${formatFeatureValue(key, value)}</div>
            <div class="feature-description">${getFeatureDescription(key)}</div>
        </div>
    `
    )
    .join("");

  resultDiv.innerHTML = `
        <div class="result-card">
            <h3><i class="fas fa-microscope"></i> Feature Analysis Results</h3>
            <div class="features-grid">
                ${featuresHtml}
            </div>
        </div>
    `;
}

// ===== Utility Functions =====
function formatFeatureName(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());
}

function formatFeatureValue(key, value) {
  if (typeof value === "number") {
    if (key.includes("score") || key.includes("sentiment")) {
      return (value * 100).toFixed(1) + "%";
    }
    return value.toFixed(3);
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
  return String(value);
}

function getFeatureIcon(key) {
  const iconMap = {
    sentiment: "fa-smile",
    credibility: "fa-star",
    emotional: "fa-heart",
    urgency: "fa-clock",
    complexity: "fa-brain",
    length: "fa-ruler",
    caps: "fa-font",
    punctuation: "fa-question",
    default: "fa-chart-bar",
  };

  for (const [pattern, icon] of Object.entries(iconMap)) {
    if (key.toLowerCase().includes(pattern)) {
      return icon;
    }
  }
  return iconMap.default;
}

function getFeatureDescription(key) {
  const descriptions = {
    sentiment_score: "Overall emotional tone of the content",
    credibility_score: "Reliability indicator based on source and content",
    emotional_manipulation: "Detection of emotionally charged language",
    urgency_indicators: "Presence of urgent or pressuring language",
    text_complexity: "Readability and linguistic sophistication",
    article_length: "Total word count of the content",
    caps_ratio: "Proportion of text in capital letters",
    punctuation_density: "Frequency of punctuation marks",
  };

  return descriptions[key] || "Analysis metric for content evaluation";
}

function setButtonLoading(button, loadingText, isLoading = true) {
  if (isLoading) {
    button.classList.add("loading");
    button.disabled = true;
    button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${loadingText}`;
  } else {
    button.classList.remove("loading");
    button.disabled = false;
    button.innerHTML = loadingText;
  }
}

function showNotification(message, type = "info") {
  // Remove existing notifications
  document.querySelectorAll(".notification").forEach((n) => n.remove());

  const notification = document.createElement("div");
  notification.className = `notification alert alert-${type}`;
  notification.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 1001;
        min-width: 300px;
        max-width: 500px;
        animation: slideInDown 0.3s ease-out;
    `;

  const icon =
    type === "success"
      ? "fa-check-circle"
      : type === "error"
      ? "fa-exclamation-circle"
      : type === "warning"
      ? "fa-exclamation-triangle"
      : "fa-info-circle";

  notification.innerHTML = `
        <i class="fas ${icon}"></i> ${message}
        <button onclick="this.parentElement.remove()" style="
            background: none; 
            border: none; 
            color: inherit; 
            float: right; 
            cursor: pointer; 
            font-size: 1.2em;
            padding: 0;
            margin-left: 10px;
        ">&times;</button>
    `;

  document.body.appendChild(notification);

  // Auto-remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.style.animation = "slideOutUp 0.3s ease-out";
      setTimeout(() => notification.remove(), 300);
    }
  }, 5000);
}

// ===== Keyboard Shortcuts =====
document.addEventListener("keydown", function (e) {
  // Ctrl/Cmd + Enter to analyze in current tab
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const activeTab = document.querySelector(".tab-content.active");
    if (activeTab) {
      const button = activeTab.querySelector(".btn");
      if (button && !button.disabled) {
        button.click();
      }
    }
    e.preventDefault();
  }

  // Ctrl/Cmd + 1/2/3 for tab switching
  if ((e.ctrlKey || e.metaKey) && ["1", "2", "3"].includes(e.key)) {
    const tabIndex = parseInt(e.key) - 1;
    const tabs = ["basic", "enhanced", "features"];
    if (tabs[tabIndex]) {
      switchTab(tabs[tabIndex]);
    }
    e.preventDefault();
  }
});

// ===== Animation Styles =====
const animationStyles = `
    @keyframes slideInDown {
        from { transform: translate(-50%, -100%); opacity: 0; }
        to { transform: translate(-50%, 0); opacity: 1; }
    }
    
    @keyframes slideOutUp {
        from { transform: translate(-50%, 0); opacity: 1; }
        to { transform: translate(-50%, -100%); opacity: 0; }
    }
`;

// Add animation styles to document
const styleSheet = document.createElement("style");
styleSheet.textContent = animationStyles;
document.head.appendChild(styleSheet);

// ===== Initialize Application =====
document.addEventListener("DOMContentLoaded", function () {
  console.log("🎯 Advanced Fake News Detector initialized");

  // Initialize theme
  initializeTheme();

  // Load available models
  loadModels();

  // Add event listeners for form submission
  document.addEventListener("keypress", function (e) {
    if (e.key === "Enter" && e.target.tagName === "TEXTAREA") {
      e.preventDefault();
      const activeTab = document.querySelector(".tab-content.active");
      if (activeTab) {
        const button = activeTab.querySelector(".btn");
        if (button && !button.disabled) {
          button.click();
        }
      }
    }
  });

  // Show welcome message
  setTimeout(() => {
    showNotification("Welcome to Advanced Fake News Detector! 🎯", "success");
  }, 1000);
});

// ===== Error Handling =====
window.addEventListener("error", function (e) {
  console.error("JavaScript error:", e.error);
  showNotification("An unexpected error occurred", "error");
});

window.addEventListener("unhandledrejection", function (e) {
  console.error("Unhandled promise rejection:", e.reason);
  showNotification("Network or server error occurred", "error");
});

// ===== Performance Monitoring =====
if ("performance" in window) {
  window.addEventListener("load", function () {
    const loadTime =
      performance.timing.loadEventEnd - performance.timing.navigationStart;
    console.log(`⚡ Page loaded in ${loadTime}ms`);
  });
}
