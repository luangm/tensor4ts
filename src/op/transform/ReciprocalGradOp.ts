import TransformOp from "./TransformOp";

export default class ReciprocalGradOp extends TransformOp {

  body(a: number, b?: number): number {
    return -1 / a / a;
  }

}