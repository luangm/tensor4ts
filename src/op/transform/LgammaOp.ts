import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";
import GammaOp from "./GammaOp";

//https://www.johndcook.com/blog/cpp_gamma/
const HALF_LOG_TWO_PI = 0.91893853320467274178032973640562;

const C_COFF = [
  1.0 / 12.0,
  -1.0 / 360.0,
  1.0 / 1260.0,
  -1.0 / 1680.0,
  1.0 / 1188.0,
  -691.0 / 360360.0,
  1.0 / 156.0,
  -3617.0 / 122400.0
];

export default class LgammaOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  static logGamma(x: number): number {
    if (x <= 0) {
      return Number.NaN;
    }

    else if (x < 12.0) {
      return Math.log(Math.abs(GammaOp.gamma(x)));
    }

    // Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252
    else {
      let z = 1.0 / (x * x);
      let sum = C_COFF[7];
      for (let i = 6; i >= 0; i--) {
        sum *= z;
        sum += C_COFF[i];
      }
      let series = sum / x;
      return (x - 0.5) * Math.log(x) - x + HALF_LOG_TWO_PI + series;
    }
  }

  body(a: number): number {
    return LgammaOp.logGamma(a);
  }

}