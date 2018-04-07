import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

// https://www.johndcook.com/blog/cpp_erf/
const a1 = 0.254829592;
const a2 = -0.284496736;
const a3 = 1.421413741;
const a4 = -1.453152027;
const a5 = 1.061405429;
const p = 0.3275911;

export default class ErfcOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    // Save the sign of x
    let sign = a >= 0 ? 1 : -1;
    a = Math.abs(a);

    // A&S formula 7.1.26
    let t = 1 / (1 + p * a);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-a * a);

    return 1 - sign * y;
  }

}