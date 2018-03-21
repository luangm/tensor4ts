import PairwiseOp from "./PairwiseOp";

export default class ModOp extends PairwiseOp {

  body(a: number, b?: number): number {
    return a % b;
  }

}