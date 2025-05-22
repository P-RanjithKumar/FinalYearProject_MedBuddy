// Dark Mode Toggle
document.addEventListener('DOMContentLoaded', function() {
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const body = document.body;

    // Check for saved preference
    const savedMode = localStorage.getItem('darkMode');
    if (savedMode === 'true') {
        body.classList.add('dark-mode');
        if (darkModeToggle) {
          darkModeToggle.querySelector('i').classList.remove('fa-moon');
          darkModeToggle.querySelector('i').classList.add('fa-sun');
        }
    }

    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            body.classList.toggle('dark-mode');
            const isDarkMode = body.classList.contains('dark-mode');

            // Update icon
            if (isDarkMode) {
                darkModeToggle.querySelector('i').classList.remove('fa-moon');
                darkModeToggle.querySelector('i').classList.add('fa-sun');
            } else {
                darkModeToggle.querySelector('i').classList.remove('fa-sun');
                darkModeToggle.querySelector('i').classList.add('fa-moon');
            }

            // Save preference
            localStorage.setItem('darkMode', isDarkMode);
        });
    }
});