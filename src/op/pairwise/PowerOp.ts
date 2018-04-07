import Tensor from "../../Tensor";
import PairwiseOp from "./PairwiseOp";

export default class PowerOp extends PairwiseOp {

  constructor(left: Tensor, right: Tensor, result: Tensor) {
    super(left, right, result);
  }

  body(a: number, b: number): number {
    return Math.pow(a, b);
  }

}