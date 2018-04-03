import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class TanGradOp extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    let sec = 1 / Math.cos(a);
    return sec * sec;
  }

}