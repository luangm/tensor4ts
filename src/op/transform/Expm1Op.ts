import TransformOp from "./TransformOp";
import Tensor from "../../Tensor";

export default class Expm1Op extends TransformOp {

  constructor(base: Tensor, result: Tensor) {
    super(base, result);
  }

  body(a: number): number {
    return Math.expm1(a);
  }

}