import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

const TWO_OVER_ROOT_PI = 2 / Math.sqrt(Math.PI);

export default class ErfGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  /**
   * 2 * exp(-x^2) / sqrt(PI)
   */
  body(a: number): number {
    return Math.exp(-a * a) * TWO_OVER_ROOT_PI;
  }

}