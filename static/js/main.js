document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const fileName = document.getElementById("file-name");
    const submitBtn = document.getElementById("submit-btn");
    const btnText = document.getElementById("btn-text");
    const btnSpinner = document.getElementById("btn-spinner");
    const uploadForm = document.getElementById("upload-form");

    const MAX_SIZE = 16 * 1024 * 1024; // 16MB
    const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/jpg"];

    function validateFile(file) {
        if (!ALLOWED_TYPES.includes(file.type)) {
            alert("Invalid file type. Please upload a PNG, JPG, or JPEG image.");
            return false;
        }
        if (file.size > MAX_SIZE) {
            alert("File is too large. Maximum size is 16MB.");
            return false;
        }
        return true;
    }

    function showPreview(file) {
        if (!validateFile(file)) return;

        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.src = e.target.result;
            fileName.textContent = file.name;
            previewContainer.style.display = "block";
            submitBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    // File input change
    fileInput.addEventListener("change", function () {
        if (this.files && this.files[0]) {
            showPreview(this.files[0]);
        }
    });

    // Drag and drop
    dropZone.addEventListener("dragover", function (e) {
        e.preventDefault();
        this.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", function () {
        this.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        this.classList.remove("drag-over");

        const files = e.dataTransfer.files;
        if (files && files[0]) {
            fileInput.files = files;
            showPreview(files[0]);
        }
    });

    // Click on drop zone opens file dialog
    dropZone.addEventListener("click", function (e) {
        if (e.target.tagName !== "LABEL" && e.target.tagName !== "INPUT") {
            fileInput.click();
        }
    });

    // Loading spinner on submit
    uploadForm.addEventListener("submit", function () {
        if (!fileInput.files || !fileInput.files[0]) {
            return;
        }
        btnText.textContent = "Analyzing...";
        btnSpinner.style.display = "inline-block";
        submitBtn.disabled = true;
    });
});
