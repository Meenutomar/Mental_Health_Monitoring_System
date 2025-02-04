// 🎤 Speech-to-Text
function startDictation(fieldId) {
    var recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.start();

    recognition.onresult = function(event) {
        document.getElementById(fieldId).value = event.results[0][0].transcript;
    };
}

// 🎵 Toggle Background Music
function toggleMusic() {
    var music = document.getElementById("bg-music");
    if (music.paused) {
        music.play();
    } else {
        music.pause();
    }
}
