pub(crate) fn number_ending(i: usize) -> &'static str {
    #[allow(clippy::match_same_arms)]
    match (i % 10, i % 100) {
        (1, 11) => "th",
        (1, _) => "st",
        (2, 12) => "th",
        (2, _) => "nd",
        (3, 13) => "th",
        (3, _) => "rd",
        (_, _) => "th",
    }
}
