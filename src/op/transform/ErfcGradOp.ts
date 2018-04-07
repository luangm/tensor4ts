import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

const NEG_TWO_OVER_ROOT_PI = -2 / Math.sqrt(Math.PI);

export default class ErfcGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  /**
   * -2 * exp(-x^2) / sqrt(PI)
   */
  body(a: number): number {
    return Math.exp(-a * a) * NEG_TWO_OVER_ROOT_PI;
  }

}