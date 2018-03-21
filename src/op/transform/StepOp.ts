import TransformOp from "./TransformOp";

export default class StepOp extends TransformOp {

  body(a: number, b?: number): number {
    return a > 0 ? 1 : 0;
  }

}