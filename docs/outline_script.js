function toggleChildChapters(event, chapterId) {
    event.preventDefault();
    const chapter = document.getElementById(chapterId);
    if (chapter.style.display === "none" || !chapter.style.display) {
        chapter.style.display = "block";
    } else {
        chapter.style.display = "none";
    }
}
