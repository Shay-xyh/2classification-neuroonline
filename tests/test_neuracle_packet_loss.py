from __future__ import annotations

import unittest

from collect.neuracle_api import ConnectState, DataServerThread


class NeuraclePacketLossTests(unittest.TestCase):
    def test_packet_discontinuity_resynchronizes_to_received_packet(self) -> None:
        server = DataServerThread(sample_rate=250, t_buffer=1.0)
        self.addCleanup(server.sock.close)
        server.state = ConnectState.RUNNING

        server.isDataPacketLost({"startTimeStamp": 100, "timeStampLength": 10})
        server.isDataPacketLost({"startTimeStamp": 120, "timeStampLength": 10})

        self.assertEqual(server.packet_loss_count, 1)
        self.assertEqual(server.lastTimestamp, 130)

        server.isDataPacketLost({"startTimeStamp": 130, "timeStampLength": 10})

        self.assertEqual(server.packet_loss_count, 1)
        self.assertEqual(server.lastTimestamp, 140)


if __name__ == "__main__":
    unittest.main()
