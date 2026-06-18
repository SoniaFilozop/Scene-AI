let lastRequestTime = 0;

function handleFile(input) {
    const file = input.files[0];
    if (!file) return;

    const now = Date.now();
    if (now - lastRequestTime < 800) return;
    lastRequestTime = now;

    showLoading(true);
    showResult(null);
    showPreview(file);

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        showLoading(false);
        showResult(data);
    })
    .catch(err => {
        showLoading(false);
        alert("Ошибка: " + err.message);
    });
}

function showPreview(file) {
    const reader = new FileReader();

    reader.onload = function(e) {
        const preview = document.getElementById("imagePreview");

        preview.innerHTML = `
            <div class="rounded-3xl overflow-hidden border border-gray-800 shadow-lg">
                <img src="${e.target.result}"
                     class="w-full max-h-[500px] object-contain bg-black">
            </div>
        `;

        preview.classList.remove("hidden");
    };

    reader.readAsDataURL(file);
}

function showLoading(state) {
    let el = document.getElementById("loading");

    if (!el) {
        el = document.createElement("div");
        el.id = "loading";
        el.className = "text-center mb-4 text-gray-400 animate-pulse";
        el.innerText = "🧠 Анализ изображения...";
        document.querySelector(".max-w-md").prepend(el);
    }

    if (state) el.classList.remove("hidden");
    else el.classList.add("hidden");
}

function showResult(data) {
    const result = document.getElementById("result");
    if (!data) {
        result.classList.add("hidden");
        return;
    }

    const icons = {
        restaurant: "🍽️",
        office: "💼",
        street: "🛣️",
        store: "🛍️",
        home: "🏠"
    };

    const icon = icons[data.scene] || "📍";

    const confidence = data.confidence ?? 0;
    const objects = data.detected_objects ?? [];

    result.innerHTML = `
        <div class="bg-gray-900 border border-gray-800 rounded-3xl p-6 shadow-xl text-center">

            <div class="text-6xl mb-2">${icon}</div>

            <div class="text-2xl font-bold capitalize">
                ${data.scene}
            </div>

            <div class="text-green-400 text-lg mt-1">
                ${confidence}%
            </div>

            <div class="h-px bg-gray-800 my-4"></div>

            <details class="text-left">
                <summary class="cursor-pointer text-gray-400">
                    Обнаруженные объекты
                </summary>

                <div class="mt-3 flex flex-wrap gap-2">
                    ${objects.length
                        ? objects.map(o => `
                            <span class="bg-gray-800 px-3 py-1 rounded-full text-sm">
                                ${o}
                            </span>
                        `).join("")
                        : `<span class="text-gray-500 text-sm">нет объектов</span>`
                    }
                </div>
            </details>

        </div>
    `;

    result.classList.remove("hidden");
}

function isIOSCameraLimited() {
    return !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia;
}


function checkDevice() {
    if (isIOSCameraLimited()) {
        console.log("iOS fallback mode: use file/capture input");
    }
}

checkDevice();