const ratings = document.querySelectorAll('.rating')
const ratingsContainer = document.querySelector('.ratings-container')
const sendBtn = document.querySelector('#send')
const panel = document.querySelector('#panel')
const feedbackTextarea = document.querySelector('#feedback')
let selectedRating = 'Satisfied'

ratingsContainer.addEventListener('click', (e) => {
    if (e.target.closest('.rating')) {
        removeActive()
        const rating = e.target.closest('.rating')
        rating.classList.add('active')
        selectedRating = rating.querySelector('small').innerText
    }
})

sendBtn.addEventListener('click', () => {
    const feedback = feedbackTextarea.value
    panel.innerHTML = `
        <div class="card shadow-sm">
            <div class="card-body text-center">
                <i class="fas fa-heart"></i>
                <strong class="d-block mb-3">Thank You!</strong>
                <strong class="d-block mb-2">Feedback: ${selectedRating}</strong>
                <p>Your feedback: ${feedback}</p>
                <p>We'll use your feedback to improve our customer support. Made by: Kumar Verma</p>
            </div>
        </div>
    `
})

function removeActive() {
    ratings.forEach(rating => rating.classList.remove('active'))
}
