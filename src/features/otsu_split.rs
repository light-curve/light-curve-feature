use crate::evaluator::*;
use crate::time_series::DataSample;
use conv::prelude::*;
use ndarray::{s, Array1, ArrayView1, Axis, Zip};
use ndarray_stats::QuantileExt;

macro_const! {
    const DOC: &'static str = r#"
Otsu threshholding algorithm

Difference of subset means, standard deviation of the lower subset, standard deviation of the upper
subset and lower-to-all observation count ratio for two subsets of magnitudes obtained by Otsu's
method split. Otsu's method is used to perform automatic thresholding. The algorithm returns a
single threshold that separate values into two classes. This threshold is determined by minimizing
intra-class intensity variance, or equivalently, by maximizing inter-class variance. There can be
more than one extremum. In this case, the algorithm returns the minimum threshold.


- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **4**

Otsu, Nobuyuki 1979. [DOI:10.1109/tsmc.1979.4310076](https://doi.org/10.1109/tsmc.1979.4310076)

Matwey Kornilov's Otsu thresholding algorithm realization was used as a reference:
    http://curl.sai.msu.ru/hg/home/matwey/domecam/file/tip/include/otsu.h
    https://ieeexplore.ieee.org/document/9170791
"#;
}

#[doc = DOC ! ()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct OtsuSplit {}

lazy_info!(
    OTSU_SPLIT_INFO,
    OtsuSplit,
    size: 4,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl OtsuSplit {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }

    pub fn threshold<'a, 'b, T>(
        ds: &'b mut DataSample<'a, T>,
    ) -> Result<(T, ArrayView1<'b, T>, ArrayView1<'b, T>), EvaluatorError>
    where
        'a: 'b,
        T: Float,
    {
        if ds.sample.len() < 2 {
            return Err(EvaluatorError::ShortTimeSeries {
                actual: ds.sample.len(),
                minimum: 2,
            });
        }

        let count = ds.sample.len();
        let countf = count.value_as::<T>().unwrap();
        let sorted = ds.get_sorted();

        if sorted.minimum() == sorted.maximum() {
            return Err(EvaluatorError::FlatTimeSeries);
        }

        // size is (count - 1)
        let cumsum1: Array1<_> = sorted
            .iter()
            .take(count - 1)
            .scan(T::zero(), |state, &m| {
                *state += m;
                Some(*state)
            })
            .collect();

        let cumsum2: Array1<_> = sorted
            .iter()
            .rev()
            .scan(T::zero(), |state, &m| {
                *state += m;
                Some(*state)
            })
            .collect();
        let cumsum2 = cumsum2.slice(s![0..count - 1; -1]);

        let amounts = Array1::range(T::one(), countf, T::one());
        let mean1 = Zip::from(&cumsum1)
            .and(&amounts)
            .map_collect(|&c, &a| c / a);
        let mean2 = Zip::from(&cumsum2)
            .and(amounts.slice(s![..;-1]))
            .map_collect(|&c, &a| c / a);

        let inter_class_variance =
            Zip::from(&amounts)
                .and(&mean1)
                .and(&mean2)
                .map_collect(|&a, &m1, &m2| {
                    let w1 = a / countf;
                    let w2 = T::one() - w1;
                    w1 * w2 * (m1 - m2).powi(2)
                });

        let index = inter_class_variance.argmax().unwrap();

        let (lower, upper) = sorted.0.view().split_at(Axis(0), index + 1);
        Ok((sorted.0[index + 1], lower, upper))
    }
}

impl FeatureNamesDescriptionsTrait for OtsuSplit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "otsu_mean_diff",
            "otsu_std_lower",
            "otsu_std_upper",
            "otsu_lower_to_all_ratio",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["difference between mean values of Otsu split subsets",
             "standard deviation for observations below the threshold given by Otsu method",
             "standard deviation for observations above the threshold given by Otsu method",
             "ratio of quantity of observations bellow the threshold given by Otsu method to quantity of all observations"]
    }
}

impl<T> FeatureEvaluator<T> for OtsuSplit
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;

        let (_, lower, upper) = Self::threshold(&mut ts.m)?;
        let mut lower: DataSample<_> = lower.into();
        let mut upper: DataSample<_> = upper.into();

        let std_lower = if lower.sample.len() == 1 {
            T::zero()
        } else {
            lower.get_std()
        };
        let mean_lower = lower.get_mean();

        let std_upper = if upper.sample.len() == 1 {
            T::zero()
        } else {
            upper.get_std()
        };
        let mean_upper = upper.get_mean();

        let mean_diff = mean_upper - mean_lower;
        let lower_to_all = lower.sample.len().value_as::<T>().unwrap() / ts.lenf();

        Ok(vec![mean_diff, std_lower, std_upper, lower_to_all])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;
    use ndarray::array;

    check_feature!(OtsuSplit);

    feature_test!(
        otsu_split,
        [OtsuSplit::new()],
        [
            0.725,
            0.012909944487358068,
            0.07071067811865482,
            0.6666666666666666
        ],
        [0.51, 0.52, 0.53, 0.54, 1.2, 1.3],
    );

    feature_test!(
        otsu_split_min_observations,
        [OtsuSplit::new()],
        [0.01, 0.0, 0.0, 0.5],
        [0.51, 0.52],
    );

    feature_test!(
        otsu_split_lower,
        [OtsuSplit::new()],
        [1.0, 0.0, 0.0, 0.25],
        [0.5, 1.5, 1.5, 1.5],
    );

    feature_test!(
        otsu_split_upper,
        [OtsuSplit::new()],
        [1.0, 0.0, 0.0, 0.75],
        [0.5, 0.5, 0.5, 1.5],
    );

    #[test]
    fn otsu_threshold() {
        let mut ds = vec![0.5, 0.5, 0.5, 1.5].into();
        let (expected_threshold, expected_lower, expected_upper) =
            (1.5, array![0.5, 0.5, 0.5], array![1.5]);
        let (actual_threshold, actual_lower, actual_upper) =
            OtsuSplit::threshold(&mut ds).expect("input is not flat");
        assert_eq!(expected_threshold, actual_threshold);
        assert_eq!(expected_lower, actual_lower);
        assert_eq!(expected_upper, actual_upper);
    }

    #[test]
    fn otsu_split_plateau() {
        let eval = OtsuSplit::new();
        let x = [1.5, 1.5, 1.5, 1.5];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(eval.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }

    /// This test remains from the issue with the previous implementation
    /// See [Issue #72] https://github.com/light-curve/light-curve-feature/issues/72
    feature_test!(
        otsu_split_small,
        [OtsuSplit::new()],
        [
            3.0221021243981205,
            0.8847146372743603,
            0.8826366394647659,
            0.507
        ],
        [
            15.00104654,
            15.00106926,
            15.00969117,
            15.01508741,
            15.02711999,
            15.030012,
            15.03287455,
            15.0372491,
            15.04885345,
            15.0495063,
            15.0504345,
            15.05288399,
            15.05800206,
            15.05935668,
            15.07334835,
            15.08106125,
            15.08366015,
            15.08953241,
            15.09150709,
            15.09427989,
            15.0956253,
            15.09706501,
            15.09837916,
            15.10175728,
            15.10523468,
            15.10895315,
            15.1109894,
            15.11712744,
            15.11737166,
            15.12069735,
            15.12173437,
            15.12314617,
            15.13369505,
            15.13704403,
            15.14412826,
            15.15560197,
            15.16277472,
            15.16983572,
            15.1701981,
            15.1715049,
            15.1729263,
            15.19134412,
            15.19194716,
            15.20081961,
            15.2036788,
            15.20861321,
            15.21882396,
            15.22150299,
            15.22260528,
            15.22952992,
            15.23546213,
            15.23612045,
            15.23829092,
            15.24657306,
            15.26199477,
            15.2622297,
            15.28662513,
            15.29248299,
            15.29255153,
            15.3042617,
            15.31485633,
            15.32892568,
            15.33715715,
            15.33850643,
            15.33975876,
            15.34210289,
            15.34535836,
            15.34555741,
            15.34916322,
            15.35304963,
            15.35852187,
            15.36502565,
            15.36913165,
            15.37000875,
            15.38627517,
            15.39528158,
            15.400915,
            15.40376576,
            15.40539647,
            15.41293478,
            15.4153605,
            15.41588379,
            15.42916044,
            15.4471012,
            15.4571105,
            15.46065927,
            15.462162,
            15.46855028,
            15.46865903,
            15.46877412,
            15.47485731,
            15.47703828,
            15.47908055,
            15.49221685,
            15.49590234,
            15.49631688,
            15.50034439,
            15.50541268,
            15.50869882,
            15.51554593,
            15.5179351,
            15.51935053,
            15.52465734,
            15.52959342,
            15.52967001,
            15.53626406,
            15.54369805,
            15.55348957,
            15.56129045,
            15.56241938,
            15.5652366,
            15.56797623,
            15.56942679,
            15.58193961,
            15.58866232,
            15.59070205,
            15.61000447,
            15.61033357,
            15.62751408,
            15.63139446,
            15.64143176,
            15.64689244,
            15.64700706,
            15.64700989,
            15.65601556,
            15.66121211,
            15.66174602,
            15.67542135,
            15.6983339,
            15.69881091,
            15.69950348,
            15.71075631,
            15.71275703,
            15.72222577,
            15.72688839,
            15.74147088,
            15.7444201,
            15.75033989,
            15.76554971,
            15.79084392,
            15.79645113,
            15.79774506,
            15.81210489,
            15.81536821,
            15.81886113,
            15.82966978,
            15.83072489,
            15.83429394,
            15.83834344,
            15.8404542,
            15.85659721,
            15.86236664,
            15.86424932,
            15.86575034,
            15.87127646,
            15.87959355,
            15.9074475,
            15.9096142,
            15.91118347,
            15.91182506,
            15.91619263,
            15.91658668,
            15.91851756,
            15.9236454,
            15.94224267,
            15.94443766,
            15.94934492,
            15.95858041,
            15.97381094,
            15.97759772,
            15.98586155,
            15.98801923,
            15.9971739,
            16.00694567,
            16.00997534,
            16.01308586,
            16.03745573,
            16.03904278,
            16.04708279,
            16.05151356,
            16.0517389,
            16.06201929,
            16.0693213,
            16.07654295,
            16.0768965,
            16.08647981,
            16.09732893,
            16.11072125,
            16.11097876,
            16.13139252,
            16.1328059,
            16.13711518,
            16.13736777,
            16.13821128,
            16.15089083,
            16.15433953,
            16.15853398,
            16.16077759,
            16.1634929,
            16.16508286,
            16.16637095,
            16.17960356,
            16.18052035,
            16.18589635,
            16.18638689,
            16.18708558,
            16.19052428,
            16.19059576,
            16.19988831,
            16.20249256,
            16.20534176,
            16.21207057,
            16.21269459,
            16.21297016,
            16.21627364,
            16.22112483,
            16.23270866,
            16.23518021,
            16.23609383,
            16.23679623,
            16.24086707,
            16.24481653,
            16.24785509,
            16.26015021,
            16.26278615,
            16.26286285,
            16.28432823,
            16.28822671,
            16.29519846,
            16.30096363,
            16.30398865,
            16.31666001,
            16.32092807,
            16.32691697,
            16.33001311,
            16.33760316,
            16.34349147,
            16.3662003,
            16.36736362,
            16.37049285,
            16.37150485,
            16.38109439,
            16.39218759,
            16.40861756,
            16.4146482,
            16.41678851,
            16.43872906,
            16.44432345,
            16.44597743,
            16.44954285,
            16.45518814,
            16.45532891,
            16.47966828,
            16.48672627,
            16.49245322,
            16.50774904,
            16.50815008,
            16.51068129,
            16.51702073,
            16.52679033,
            16.52825423,
            16.52951941,
            16.52956343,
            16.53285886,
            16.5450217,
            16.54706444,
            16.54778882,
            16.55707041,
            16.55993501,
            16.55995873,
            16.56113737,
            16.59395944,
            16.59458179,
            16.59769974,
            16.5988398,
            16.60082435,
            16.60396362,
            16.61054663,
            16.6146126,
            16.61657895,
            16.61811921,
            16.6478395,
            16.66003712,
            16.66415318,
            16.67181074,
            16.67207775,
            16.69621478,
            16.69641226,
            16.69743813,
            16.70206573,
            16.70784012,
            16.71593619,
            16.71864147,
            16.72422574,
            16.72430892,
            16.72530067,
            16.73078696,
            16.73692713,
            16.73769023,
            16.74572851,
            16.75000123,
            16.76606042,
            16.77456858,
            16.77589189,
            16.79772983,
            16.79775834,
            16.8047019,
            16.81602013,
            16.8173781,
            16.82301584,
            16.82553838,
            16.83695046,
            16.84482086,
            16.84503487,
            16.85402627,
            16.85858705,
            16.87413031,
            16.87518553,
            16.87975244,
            16.91546446,
            16.91797901,
            16.93156949,
            16.93483658,
            16.94346599,
            16.94628376,
            16.94997188,
            16.95686191,
            16.95810871,
            16.96219862,
            16.96294304,
            16.98095696,
            16.98128684,
            16.98203352,
            16.98441996,
            16.99316755,
            17.00348224,
            17.00480861,
            17.00651156,
            17.01858104,
            17.02214734,
            17.02713784,
            17.03210655,
            17.03218351,
            17.03556691,
            17.04605484,
            17.05006724,
            17.0575355,
            17.05766769,
            17.06656748,
            17.06956214,
            17.07059728,
            17.07196866,
            17.0796932,
            17.08159995,
            17.0836899,
            17.09730999,
            17.10042439,
            17.10186961,
            17.10307436,
            17.11565234,
            17.13126359,
            17.13126455,
            17.14052657,
            17.1406197,
            17.14288315,
            17.14538234,
            17.14889518,
            17.15016388,
            17.16083342,
            17.16251743,
            17.16433232,
            17.16697722,
            17.16841721,
            17.17862999,
            17.18425145,
            17.18816043,
            17.1930866,
            17.19463727,
            17.1987289,
            17.20199902,
            17.2066429,
            17.20708141,
            17.21495987,
            17.21574978,
            17.21621172,
            17.22096572,
            17.22284283,
            17.22321565,
            17.22558065,
            17.23359232,
            17.2410273,
            17.24340897,
            17.24434373,
            17.24817053,
            17.25747381,
            17.25877584,
            17.26373053,
            17.30492656,
            17.30852279,
            17.31012944,
            17.31083011,
            17.32254488,
            17.33428481,
            17.34606466,
            17.35060712,
            17.35581652,
            17.36144547,
            17.36339785,
            17.36342751,
            17.36843318,
            17.36877886,
            17.38345505,
            17.3845665,
            17.38920315,
            17.40867332,
            17.42713866,
            17.44441279,
            17.4469125,
            17.44741069,
            17.45940296,
            17.46357168,
            17.48913584,
            17.49377459,
            17.50907919,
            17.5124283,
            17.51752791,
            17.52410395,
            17.5248082,
            17.53387169,
            17.53640697,
            17.55505788,
            17.55980685,
            17.56107292,
            17.56980835,
            17.58030418,
            17.58245504,
            17.59336361,
            17.6002285,
            17.60139244,
            17.60712275,
            17.61260012,
            17.61521863,
            17.63220503,
            17.63248683,
            17.64012773,
            17.64548846,
            17.6470224,
            17.65403929,
            17.66771884,
            17.70623741,
            17.7078488,
            17.71738678,
            17.7231846,
            17.72530969,
            17.72766323,
            17.7334287,
            17.73397484,
            17.73755063,
            17.75086675,
            17.7547237,
            17.76544875,
            17.76769614,
            17.77406138,
            17.77627878,
            17.77718064,
            17.78167228,
            17.78339974,
            17.78554073,
            17.78990295,
            17.7940748,
            17.79630385,
            17.79653211,
            17.79872257,
            17.79976075,
            17.80851554,
            17.81281985,
            17.82811226,
            17.83564671,
            17.85164556,
            17.8526896,
            17.85630178,
            17.85643776,
            17.85664229,
            17.87483423,
            17.88032985,
            17.8847091,
            17.89254347,
            17.89319254,
            17.8947859,
            17.89980579,
            17.90282618,
            17.9047401,
            17.91391192,
            17.91407084,
            17.92620589,
            17.92753991,
            17.93811088,
            17.93963212,
            17.94834263,
            17.95356539,
            17.95376595,
            17.95547935,
            17.95644587,
            17.96293596,
            17.96685718,
            17.98017765,
            17.98023493,
            17.98616431,
            17.9961507,
            18.00104972,
            18.00355035,
            18.01075525,
            18.01154481,
            18.04181171,
            18.04222718,
            18.0490931,
            18.04945853,
            18.05957351,
            18.06520277,
            18.06820566,
            18.07498348,
            18.08716206,
            18.08889559,
            18.09292008,
            18.09565348,
            18.10514133,
            18.10641637,
            18.119633,
            18.12191468,
            18.1316934,
            18.13752792,
            18.13769167,
            18.14348342,
            18.14481823,
            18.15783363,
            18.16870196,
            18.16998118,
            18.17232862,
            18.18325111,
            18.19160468,
            18.19203733,
            18.19320552,
            18.19929096,
            18.2054243,
            18.21013247,
            18.21730702,
            18.2190333,
            18.22440942,
            18.22905215,
            18.23474153,
            18.23857562,
            18.24023793,
            18.24650875,
            18.2627136,
            18.26428142,
            18.26968502,
            18.27794903,
            18.27809564,
            18.2833956,
            18.2872506,
            18.29757639,
            18.29855147,
            18.29925556,
            18.30111654,
            18.30201362,
            18.33327029,
            18.34297711,
            18.35537461,
            18.36307225,
            18.3794421,
            18.38358458,
            18.3854774,
            18.39505015,
            18.39833502,
            18.3993728,
            18.40249828,
            18.40604645,
            18.41866553,
            18.42895297,
            18.42982613,
            18.43163216,
            18.44541519,
            18.44614755,
            18.44849159,
            18.45190839,
            18.4581089,
            18.45998373,
            18.46211031,
            18.48945199,
            18.49010113,
            18.49706609,
            18.500898,
            18.50398815,
            18.50513392,
            18.50867204,
            18.52087124,
            18.52669082,
            18.52728325,
            18.54025133,
            18.54940778,
            18.55244286,
            18.56810144,
            18.57416966,
            18.57667715,
            18.58889609,
            18.59590613,
            18.59604076,
            18.60001642,
            18.60424078,
            18.61614517,
            18.62303988,
            18.63223943,
            18.66037311,
            18.66131582,
            18.66707227,
            18.67048498,
            18.67388207,
            18.67669114,
            18.67796576,
            18.68735779,
            18.6912832,
            18.69319326,
            18.69986254,
            18.70448452,
            18.70468571,
            18.70755531,
            18.70917929,
            18.71687023,
            18.72086905,
            18.72258005,
            18.73925651,
            18.73991641,
            18.744851,
            18.74824368,
            18.74841145,
            18.74977187,
            18.75812555,
            18.76417356,
            18.76894445,
            18.76913123,
            18.77046788,
            18.77254981,
            18.78267488,
            18.78332167,
            18.78575257,
            18.78779903,
            18.8019313,
            18.81058345,
            18.82157965,
            18.8245087,
            18.82473682,
            18.82617739,
            18.82795295,
            18.82895636,
            18.83123224,
            18.83660424,
            18.83710841,
            18.8438842,
            18.84438707,
            18.84472482,
            18.84688378,
            18.85433083,
            18.86479137,
            18.87068316,
            18.87157417,
            18.87876198,
            18.8920737,
            18.90244542,
            18.91375787,
            18.91425397,
            18.91810086,
            18.92292768,
            18.94333489,
            18.95472567,
            18.9564486,
            18.95672788,
            18.96125129,
            18.96603273,
            18.98372334,
            18.99467123,
            19.00713442,
            19.00830525,
            19.0330598,
            19.05999941,
            19.06916102,
            19.09152247,
            19.09467652,
            19.10173826,
            19.10432448,
            19.11036723,
            19.1224538,
            19.1241954,
            19.13042442,
            19.13142649,
            19.13487256,
            19.1409495,
            19.16140503,
            19.16320883,
            19.16676804,
            19.17453578,
            19.17640517,
            19.18131865,
            19.18398847,
            19.19234251,
            19.19668256,
            19.22106795,
            19.22978191,
            19.2335749,
            19.23492534,
            19.24590553,
            19.24627484,
            19.24984414,
            19.25005903,
            19.25435241,
            19.25940367,
            19.26789478,
            19.27604166,
            19.27676616,
            19.27704976,
            19.2801813,
            19.28966457,
            19.30238371,
            19.30247903,
            19.30727785,
            19.32349707,
            19.32632274,
            19.33039161,
            19.33798282,
            19.34267805,
            19.3430206,
            19.34498022,
            19.34559166,
            19.35180654,
            19.36157903,
            19.3646987,
            19.36708679,
            19.37000547,
            19.37524593,
            19.3753111,
            19.38214156,
            19.39030378,
            19.3979723,
            19.39964463,
            19.40152231,
            19.40704383,
            19.4157907,
            19.41945241,
            19.422468,
            19.42287617,
            19.43921178,
            19.44893117,
            19.45967608,
            19.46536022,
            19.46762294,
            19.46812933,
            19.46982002,
            19.47137666,
            19.47143284,
            19.48407085,
            19.49469539,
            19.50339817,
            19.51263658,
            19.51631218,
            19.52103402,
            19.55941907,
            19.56424064,
            19.57015748,
            19.57028957,
            19.57262724,
            19.57360724,
            19.57464529,
            19.58574515,
            19.58900418,
            19.59175672,
            19.59872475,
            19.60581934,
            19.62234763,
            19.63289258,
            19.64785167,
            19.65384,
            19.65735152,
            19.65971363,
            19.66453668,
            19.66624458,
            19.67081754,
            19.67322705,
            19.67403863,
            19.68788276,
            19.68882813,
            19.6961061,
            19.6984687,
            19.70542186,
            19.73049432,
            19.73941291,
            19.73946543,
            19.74609278,
            19.74787845,
            19.75117956,
            19.77949858,
            19.78190763,
            19.78487757,
            19.79721439,
            19.79775435,
            19.80075344,
            19.80720428,
            19.81657245,
            19.821489,
            19.82280706,
            19.82688335,
            19.83171944,
            19.8348221,
            19.83675263,
            19.84088208,
            19.84487043,
            19.8454604,
            19.84667878,
            19.84809565,
            19.8660879,
            19.87566406,
            19.87894814,
            19.88880064,
            19.89791101,
            19.8989651,
            19.9016721,
            19.90330613,
            19.90374921,
            19.91144282,
            19.93430511,
            19.94056132,
            19.95710488,
            19.9578813,
            19.96535738,
            19.98584495,
            19.99331944,
            19.99356309,
            20.01476116,
            20.03058093,
            20.05156771,
            20.05950843,
            20.05971075,
            20.06133512,
            20.07373227,
            20.07402846,
            20.09813057,
            20.10301666,
            20.1092664,
            20.11245613,
            20.12791466,
            20.12898184,
            20.13733742,
            20.14241652,
            20.14385212,
            20.14471866,
            20.14724621,
            20.15170649,
            20.16422519,
            20.16841503,
            20.17561705,
            20.1763825,
            20.18092693,
            20.18133814,
            20.18201528,
            20.18703049,
            20.18730094,
            20.18867321,
            20.1950089,
            20.20909856,
            20.2105845,
            20.21092734,
            20.22548591,
            20.23218896,
            20.25846572,
            20.2604402,
            20.26897594,
            20.28990476,
            20.29920983,
            20.30058805,
            20.32584462,
            20.33446743,
            20.33793506,
            20.34218482,
            20.343146,
            20.3464133,
            20.34945798,
            20.35155897,
            20.3641261,
            20.37538635,
            20.38147009,
            20.38328091,
            20.3890608,
            20.39151157,
            20.3915826,
            20.39502747,
            20.39701806,
            20.41950361,
            20.42173387,
            20.42210634,
            20.42878829,
            20.43925758,
            20.4409022,
            20.45132957,
            20.45256747,
            20.46843718,
            20.47294842,
            20.48368807,
            20.484193,
            20.48426286,
            20.49023467,
            20.50357108,
            20.50701133,
            20.5102,
            20.51518666,
            20.51744445,
            20.51969116,
            20.52137803,
            20.52160709,
            20.52475793,
            20.52489685,
            20.53170204,
            20.53504536,
            20.53517507,
            20.55170731,
            20.55415941,
            20.55704755,
            20.55844937,
            20.56368593,
            20.56521336,
            20.56675036,
            20.57798401,
            20.5785649,
            20.58238026,
            20.58786236,
            20.58809458,
            20.58823895,
            20.59477295,
            20.59809528,
            20.60294206,
            20.63221882,
            20.63259018,
            20.66799698,
            20.66901888,
            20.67833101,
            20.6838271,
            20.68493378,
            20.69054241,
            20.69600042,
            20.70141175,
            20.71124921,
            20.7171501,
            20.72487316,
            20.72817207,
            20.73521519,
            20.73747819,
            20.74211153,
            20.7466377,
            20.75100016,
            20.75977894,
            20.76235236,
            20.76446343,
            20.76481232,
            20.76551931,
            20.76652992,
            20.76779783,
            20.77228395,
            20.77522593,
            20.78526147,
            20.78740685,
            20.79858321,
            20.80420874,
            20.81003496,
            20.81204574,
            20.81394788,
            20.83027294,
            20.83732721,
            20.84643978,
            20.8509044,
            20.8554153,
            20.8595742,
            20.86038674,
            20.86142595,
            20.87082373,
            20.87377279,
            20.87979219,
            20.88020776,
            20.88048672,
            20.88303634,
            20.90727124,
            20.90969919,
            20.91356648,
            20.91926784,
            20.92428836,
            20.9320429,
            20.93422247,
            20.94073689,
            20.94527978,
            20.95897692,
            20.9601938,
            20.96879124,
            20.98084027,
            20.98605297,
        ],
    );
}
