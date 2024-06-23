document.addEventListener("DOMContentLoaded", function() {
    // Handle the Home link click
    const homeLink = document.querySelector(".home-link");
    homeLink.addEventListener("click", function(event) {
        event.preventDefault();
        renderHomeBanner();
    });

    function renderHomeBanner() {
        const bannerContainer = document.getElementById("banner-container");
        bannerContainer.innerHTML = `
            <div class="banner">
                <div class="slider" style="--quantity: 4">
                    <div class="item" style="--position: 1"> 
                        <img class="profile" src="./images/Screenshot 2024-06-23 at 9.25.41 AM.png" alt=""> 
                        <p class="name">Aung Nanda Oo</p>
                    </div>
                    <div class="item" style="--position: 2"> 
                        <img class="profile" src="./images/Kyle.jpg" alt=""> 
                        <p class="name">Kyle</p>
                    </div>
                    <div class="item" style="--position: 3"> 
                        <img class="profile" src="./images/Screenshot 2024-06-23 at 10.47.02 AM.png" alt=""> 
                        <p class="name">Angelo</p>
                    </div>
                    <div class="item" style="--position: 4"> 
                        <img class="profile" src="./images/Screenshot 2024-06-23 at 10.51.24 AM.png" alt=""> 
                        <p class="name">Maya</p>
                    </div>
                    <div class="content">
                        <div class="model"> </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Initial render of the home banner
    renderHomeBanner();
});
