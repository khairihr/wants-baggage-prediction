// script.js

document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');

    form.addEventListener('submit', function(event) {
        // Validasi input
        let isValid = true;
        const inputs = form.querySelectorAll('input, select');

        for (const input of inputs) {
            if (!input.value) {
                isValid = false;
                alert(`Harap isi kolom ${input.labels[0].textContent}`); // Tampilkan pesan error
                break; // Hentikan validasi jika ada input kosong
            }
        }

        if (!isValid) {
            event.preventDefault(); // Cegah formulir dikirim jika tidak valid
        }
    });
});
