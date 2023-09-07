/// Return a suffix for a number, like "st", "nd", or "th".
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        assert_eq!(number_ending(0), "th");
        assert_eq!(number_ending(1), "st");
        assert_eq!(number_ending(2), "nd");
        assert_eq!(number_ending(3), "rd");
        assert_eq!(number_ending(4), "th");
        assert_eq!(number_ending(5), "th");
        assert_eq!(number_ending(6), "th");
        assert_eq!(number_ending(7), "th");
        assert_eq!(number_ending(8), "th");
        assert_eq!(number_ending(9), "th");
        assert_eq!(number_ending(10), "th");
        assert_eq!(number_ending(11), "th");
        assert_eq!(number_ending(12), "th");
        assert_eq!(number_ending(13), "th");
        assert_eq!(number_ending(14), "th");
        assert_eq!(number_ending(15), "th");
        assert_eq!(number_ending(16), "th");
        assert_eq!(number_ending(17), "th");
        assert_eq!(number_ending(18), "th");
        assert_eq!(number_ending(19), "th");
        assert_eq!(number_ending(20), "th");
        assert_eq!(number_ending(21), "st");
        assert_eq!(number_ending(22), "nd");
        assert_eq!(number_ending(23), "rd");
        assert_eq!(number_ending(24), "th");
        assert_eq!(number_ending(25), "th");
        assert_eq!(number_ending(100), "th");
        assert_eq!(number_ending(101), "st");
        assert_eq!(number_ending(102), "nd");
    }
}
