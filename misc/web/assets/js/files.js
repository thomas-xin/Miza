(() => {
    
    function setupFileInput() {
        /** @type {HTMLInputElement} */
        const fileDropZone = document.querySelector('section.file-upload');
        if (!fileDropZone) {
            return;
        }
        /** @type {HTMLInputElement} */
        const label = fileDropZone.querySelector('.file-input');
        if (!label) {
            return;
        }

        let textElem = label.querySelector('.filename-text');
        let inputElem = label.querySelector('input'); 

        if (!textElem || !inputElem) {
            console.warn("This would have matched, but it's missing required elements: ", label);
            return;
        }

        let defaultText = textElem.innerText;
        if (inputElem.files && inputElem.files[0]) {
            textElem.innerText = inputElem.files[0].name;
        }

        function onChange() {
            if (inputElem.files && inputElem.files[0]) {
                textElem.innerText = inputElem.files[0].name;
            } else {
                textElem.innerText = defaultText;
            }
            label.classList.remove('pulse');
            setTimeout(() => label.classList.add('pulse'), 5);
        }

        inputElem.addEventListener('change', onChange)

        fileDropZone.addEventListener('drop', (e) => {
            console.log(e);

            e.preventDefault();

            // pretty simple -- but not for IE :(
            inputElem.files = e.dataTransfer.files;

            // If you want to use some of the dropped files
            const dT = new DataTransfer();
            dT.items.add(e.dataTransfer.files[0]);
            inputElem.files = dT.files;


            onChange();
        });


        fileDropZone.addEventListener("dragover", (e) => {
            const fileItems = [...e.dataTransfer.items].filter(
                (item) => item.kind === "file",
            );
            if (fileItems.length > 0) {
                e.preventDefault();
                e.dataTransfer.dropEffect = "copy";
            }
        });

        // if this is on fileDropZone you have to click in
        // it's annoying
        window.addEventListener('paste', (e) => {
            const clipboardFiles = e.clipboardData?.files;
            
            // Check if there's actually a file in the clipboard (ignores text pastes)
            if (clipboardFiles && clipboardFiles.length > 0) {
                inputElem.files = clipboardFiles;
                
                onChange();

                // Clear any previous errors to show it was successful
                errorDiv.classList.add('hidden');
            }
        });
    }

    function populatePreviewSection(filename, url, mime, size) {

        function formatBytes(bytes) {
            if (bytes <= 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }
        
        let previewSection = document.querySelector('section.file-preview');
        let h1 = previewSection.querySelector('h1');
        let permalink = previewSection.querySelector('.permalink');
        let mediaPreviewHolder = previewSection.querySelector('.media-preview-holder');
        let metadataElem = previewSection.querySelector('.metadata');
        let viewRawBtn = previewSection.querySelector('#view-raw');
        let dlButton = previewSection.querySelector('#download');

        h1.innerText = '“' + filename + '”';
        permalink.href = url;
        permalink.innerText = url;

        if (mime.startsWith('image/')) {
            let previewImg = document.createElement('img');
            previewImg.src = url;
            previewImg.alt = filename;
            mediaPreviewHolder.appendChild(previewImg);
        } else if (mime.startsWith('video/')) {
            let previewVideo = document.createElement('video');
            previewVideo.src = url;
            previewVideo.controls = true;
            mediaPreviewHolder.appendChild(previewVideo);
        } else if (mime.startsWith('audio/')) {
            let previewVideo = document.createElement('audio');
            previewVideo.src = url;
            previewVideo.controls = true;
            mediaPreviewHolder.appendChild(previewVideo);
        }

        metadataElem.innerText = `${mime}, ${formatBytes(size)}`

        viewRawBtn.href = url;
        const downloadUrl = new URL(url, window.location);
        downloadUrl.searchParams.set('download', '1');
        dlButton.href = downloadUrl.href;


        const previewUrl = new URL(window.location);
        previewUrl.searchParams.set('url', url);

        if (new URLSearchParams(window.location.search).get('url') !== url) {
            history.pushState({}, "", previewUrl.toString());
        }
    }

    function uploadFile(persistent, file) {
        return new Promise((res, rej) => {
            const uploadUrl = persistent ? `https://mizabot.xyz/upload` : `https://api.mizabot.xyz/upload`;
            let parsedUrl = new URL(uploadUrl);
            parsedUrl.searchParams.set('filename', file.name);

            const formData = new FormData();
            formData.append('file', file);

            // Use XMLHttpRequest to track upload progress accurately
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = Math.round((e.loaded / e.total) * 100);
                    let uploadSection = document.querySelector('section.file-upload');
                    let progressH2 = document.getElementById('uploadProgressH2');
                    uploadSection.style.setProperty('--progress', progress);
                    progressH2.innerText = `${progress}% complete`;
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    // Success - redirect to preview page
                    res(xhr.responseText.trim());
                } else {
                    // HTTP connection succeeded, but returned an error status code
                    rej(`Upload failed: HTTP ${xhr.status} - ${xhr.statusText}`)
                }
            });

            xhr.addEventListener('error', () => {
                rej("A network error occurred during upload.");
            });

            xhr.open('POST', parsedUrl.toString(), true);
            xhr.send(formData);
        });
    }

    function setupFormSubmit() {
        /** @type {HTMLFormElement} */
        let form = document.querySelector('section.file-upload > form');
        let uploadSection = document.querySelector('section.file-upload');
        let uploadH1 = document.getElementById('uploadH1');
        let progressH2 = document.getElementById('uploadProgressH2');

        let previewSection = document.querySelector('section.file-preview');

        form.addEventListener('submit', e => {
            e.preventDefault();

            (async() => {
                let data = new FormData(form);
                /** @type {File} */
                let file = data.get('file');
                let persistent = data.get('keep') === 'true';
                let uploading = true;

                uploadSection.classList.add('uploading');
                window.onbeforeunload = () => {
                    if (uploading) { return 'Are you sure you want to cancel your upload?' }
                    return null;
                };

                uploadH1.innerText = 'File uploading...';
                let resultUrl = null;

                try {
                    resultUrl = await uploadFile(persistent, file);
                } catch (e) {
                    uploadH1.innerText = 'Upload failed';
                    progressH2.innerText = e;
                    uploading = false;
                    return;
                }

                uploadH1.innerText = 'File uploaded.';
                uploadSection.style.setProperty('--progress', 100);
                progressH2.innerText = `100% complete`;

                uploadSection.classList.add('doneUploading');
                uploading = false;

                populatePreviewSection(
                    file.name,
                    resultUrl,
                    file.type,
                    file.size
                )

                previewSection.classList.add('visible');
            })();
        })
    }
    

    async function runSetupFuncs() {
        setupFileInput();
        setupFormSubmit();

        let viewUrl = new URLSearchParams(window.location.search).get('url');
        if (viewUrl) {
            let form = document.querySelector('section.file-upload > form');
            let uploadSection = document.querySelector('section.file-upload');
            let uploadH1 = document.getElementById('uploadH1');
            let progressH2 = document.getElementById('uploadProgressH2');

            let previewSection = document.querySelector('section.file-preview');

            uploadSection.classList.add('uploading');

            uploadH1.innerText = 'Loading preview...';
            uploadSection.style.setProperty('--progress', 100);
            progressH2.innerText = ``;

			const parsedUrl = new URL(viewUrl, window.location.origin);
			const baseDomain = "mizabot.xyz";

			if (
                parsedUrl.hostname !== baseDomain &&
                !parsedUrl.hostname.endsWith('.' + baseDomain)
            ) {
                uploadH1.innerText = 'Error loading preview';
				progressH2.innerText = "Previewing files from unrecognized domains is disabled.";
                return;
			}

			const response = await fetch(parsedUrl.href, { method: 'GET' });

			if (!response.ok) {
                uploadH1.innerText = 'Error loading preview';
				progressH2.innerText = `File not found or accessible. HTTP ${response.status}`;
                return;
			}

			const contentType = response.headers.get('Content-Type') || 'application/octet-stream';
			const contentLength = response.headers.get('Content-Length') || '0';
			const contentDisposition = response.headers.get('Content-Disposition') || '';

			// Use the new helper, and fallback to the URL path if the header is missing 
			let filename = decodeURIComponent(contentDisposition.replace('inline; filename=', ''));
			if (!filename) {
				filename = parsedUrl.pathname.split('/').pop() || 'Unknown File';
			}

			// Calculate human-readable size
			const bytes = parseInt(contentLength);

            populatePreviewSection(
                filename,
                viewUrl,
                contentType,
                bytes
            );

            previewSection.classList.add('visible');
        }


    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', runSetupFuncs);
    } else {
        runSetupFuncs();
    }
})();